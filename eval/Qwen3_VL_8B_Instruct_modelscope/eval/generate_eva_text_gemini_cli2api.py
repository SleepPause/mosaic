#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 OpenAI 兼容模式通过 Gemini Manager 代理调用 Gemini API（多线程并发版本，支持 thinking）
文件名：图片名_策略名_序号_gemini_时间戳.txt
"""
import time
import re
import logging
import random
import signal
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from PIL import Image
import prompt_strategies
import evaluation_templates
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ========= 用户配置区域 =========
# Gemini Manager 代理配置（显式配置）
GEMINI_MANAGER_PORT = "4000"
GEMINI_MANAGER_PASSWORD = "123456"  # 请修改为你的密码
GEMINI_MANAGER_BASE_URL = f"http://localhost:{GEMINI_MANAGER_PORT}/v1"

# 并发线程数（每天 1500 次限制，可以根据需要调整）
NUM_THREADS = 1

# Gemini 模型配置（循环队列）
MODELS = ["gemini-2.5-flash"]

# 使用脚本所在目录作为基准
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"
OUT_DIR = BASE_DIR / "eval_text"

# 定义要测试的策略集合
STRATEGIES = {
    "BreakingBad": prompt_strategies.PROMPT_BREAKING_BAD,
    "SafetyAuditor": prompt_strategies.PROMPT_SAFETY_AUDITOR,
    "Historian": prompt_strategies.PROMPT_HISTORIAN,
    "JSON": prompt_strategies.PROMPT_JSON_FORMAT,
    "ArtCritic": prompt_strategies.PROMPT_ART_CRITIC,
}

# 每个策略重复测试的次数
REPEAT_TIMES = 6

# 线程失败阈值：连续失败次数超过此值，线程退出
THREAD_FAILURE_THRESHOLD = len(MODELS) * 3

# 速率限制配置（保守配置，避免触发限制）
SUCCESS_SLEEP_MIN = 5  # 成功后最少等待 15 秒
SUCCESS_SLEEP_MAX = 10  # 成功后最多等待 20 秒
FAILURE_SLEEP = 60      # 失败后等待 60 秒
# =================================

# 使用 StrongREJECT 模板
EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT

# 线程安全锁
print_lock = threading.Lock()
stats_lock = threading.Lock()
active_threads_lock = threading.Lock()

# 统计计数器
stats = {
    'completed': 0,    # 成功完成的任务
    'skipped': 0,      # 跳过的任务（已存在）
    'failed': 0        # 失败的任务（重新入队）
}

# 活跃线程计数器
active_threads = 0

# 优雅退出标志
stop_flag = False

# 日志系统
logger = None  # 将在 main 函数中初始化


def setup_logging(log_dir: Path):
    """
    配置日志系统：详细日志写入文件，控制台只显示关键信息
    """
    global logger
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    log_file = log_dir / f"eval_gemini_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logger = logging.getLogger('GeminiOpenAIEval')
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器 - 记录所有详细信息
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)
    
    print(f"📋 日志文件: {log_file}\n")
    
    return log_file


def clean_reasoning_content(reasoning: str) -> str:
    """
    清理 reasoning_content 中的重复 JSON 代码块
    针对 JSON 策略时 Gemini API 返回的冗余思考过程
    
    处理两种情况：
    1. 完整的重复 JSON 块：```json ... ```
    2. 不完整的、被截断的 JSON 块：```json ... (没有结束```)
    
    Args:
        reasoning: 原始 reasoning 内容
    
    Returns:
        清理后的 reasoning 内容
    """
    if not reasoning or len(reasoning) < 500:
        return reasoning
    
    import re
    
    # 1. 首先检测是否有未闭合的 JSON 代码块（从```json开始但没有```结束）
    # 查找最后一个```json的位置
    last_json_start = reasoning.rfind('```json')
    
    if last_json_start != -1:
        # 检查在此位置之后是否有闭合的```
        after_last_json = reasoning[last_json_start + 7:]  # 跳过```json
        
        # 查找下一个```
        next_close = after_last_json.find('```')
        
        if next_close == -1:
            # 没有找到闭合标记，说明是截断的JSON块
            # 检查这个未闭合的块是否在文件末尾附近（最后3000字符内）
            if len(reasoning) - last_json_start < 3000:
                # 很可能是截断的重复内容，移除这个不完整的块
                logger.debug(f"Detected incomplete JSON block at position {last_json_start}, removing truncated content")
                reasoning = reasoning[:last_json_start].rstrip() + "\n\n[不完整的 JSON 块已移除（可能被截断）]\n"
                return reasoning
    
    # 2. 处理完整的重复 JSON 块
    # 查找所有完整的 JSON 代码块 (```json ... ```)
    json_blocks = re.findall(r'```json\s*\n(.*?)\n```', reasoning, re.DOTALL)
    
    if len(json_blocks) <= 1:
        # 没有重复,直接返回
        return reasoning
    
    # 检测重复的 JSON 块
    seen_blocks = set()
    duplicate_indices = []
    
    for i, block in enumerate(json_blocks):
        # 简化 JSON 块用于比较（去除空格换行）
        normalized = re.sub(r'\s+', '', block)
        
        if normalized in seen_blocks:
            duplicate_indices.append(i)
        else:
            seen_blocks.add(normalized)
    
    # 如果有重复,移除重复的块
    if duplicate_indices:
        # 分割 reasoning 按 JSON 块
        parts = re.split(r'(```json\s*\n.*?\n```)', reasoning, flags=re.DOTALL)
        
        # 标记要移除的块
        json_block_counter = 0
        filtered_parts = []
        
        for part in parts:
            if re.match(r'```json\s*\n.*?\n```', part, re.DOTALL):
                # 这是一个 JSON 块
                if json_block_counter not in duplicate_indices:
                    filtered_parts.append(part)
                else:
                    # 跳过重复块,但保留一个占位符
                    filtered_parts.append("\n[重复的 JSON 块已移除]\n")
                json_block_counter += 1
            else:
                # 保留非 JSON 部分
                filtered_parts.append(part)
        
        cleaned = ''.join(filtered_parts)
        
        # 记录清理信息
        removed_count = len(duplicate_indices)
        logger.debug(f"Cleaned {removed_count} duplicate JSON blocks from reasoning_content")
        
        return cleaned
    
    return reasoning


def log_and_print(message: str, level: str = 'info', console: bool = False):
    """
    智能日志输出：
    - 详细信息记录到日志文件
    - 重要信息同时显示在控制台
    """
    if logger is None:
        return
    
    # 写入日志文件
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)
    
    # 如果需要，在控制台显示（使用 tqdm.write 避免干扰进度条）
    if console:
        with print_lock:
            tqdm.write(message)


def get_client():
    """获取 OpenAI 客户端（连接到 Gemini Manager）"""
    return OpenAI(
        base_url=GEMINI_MANAGER_BASE_URL,
        api_key=GEMINI_MANAGER_PASSWORD
    )


def image_to_base64(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_image_mime_type(image_path: Path) -> str:
    """获取图片 MIME 类型"""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/jpeg"


def safe_file_name(name: str) -> str:
    """去掉特殊符号"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_response(client, text_prompt: str, image_path: Path, model: str, thread_id: int) -> tuple:
    """
    使用 OpenAI 兼容模式调用 Gemini（通过 Gemini Manager 代理）
    
    Args:
        client: OpenAI 客户端
        text_prompt: 文本提示词
        image_path: 图片路径
        model: 模型名称
        thread_id: 线程 ID
    
    Returns:
        (answer, should_switch_model, reasoning): 
            answer - API响应或错误信息
            should_switch_model - 是否应该切换模型
            reasoning - 思考过程（thinking process）
    """
    try:
        # 准备图片数据
        base64_image = image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        # 构建消息（OpenAI Vision 格式）
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        
        # 提取答案
        answer = response.choices[0].message.content
        
        # 提取 thinking（Gemini Manager 的自定义字段）
        reasoning = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        
        # 检查响应是否为空
        if not answer:
            return "API Error: Empty response from API", False, ""
        
        # 检查是否是代理的错误消息
        proxy_error_indicators = [
            "所有API密钥均请求失败",
            "具体错误请查看轮询日志",
            "All API keys failed",
            "API key轮询失败"
        ]
        
        if any(indicator in answer for indicator in proxy_error_indicators):
            log_and_print(f"[Thread-{thread_id}] 检测到代理错误消息: {answer[:100]}", level='warning')
            return f"API Error: Proxy reported - {answer[:200]}", True, ""
        
        return answer, False, reasoning  # 成功
        
    except Exception as e:
        error_str = str(e)
        # 记录到日志文件
        log_and_print(f"[Thread-{thread_id}] API错误 ({model}): {error_str}", level='warning')
        
        # 检查是否是需要切换模型的错误
        should_switch = any(code in error_str for code in [
            "429", "503", "500", "403", "400", "404", "504",
            "timeout", "timed out", "Timeout", "RESOURCE_EXHAUSTED"
        ])
        
        return f"API Error: {error_str}", should_switch, ""


def process_with_thread(thread_id, task_queue, pbar):
    """
    每个线程对应的工作函数
    """
    global active_threads
    
    client = get_client()
    current_model_index = 0
    consecutive_failures = 0
    
    # 线程启动 - 增加活跃线程计数
    with active_threads_lock:
        active_threads += 1
        current_active = active_threads
    
    log_and_print(f"🔑 Thread-{thread_id} 启动（当前活跃线程: {current_active}）", console=True)
    
    while True:
        # 检查是否需要优雅退出
        if stop_flag:
            log_and_print(f"[Thread-{thread_id}] 接收到停止信号，优雅退出", console=True)
            break
        
        task = None
        task_acquired = False
        
        try:
            # 检查是否达到失败阈值
            if consecutive_failures >= THREAD_FAILURE_THRESHOLD:
                log_and_print(f"💀 [Thread-{thread_id}] 连续失败 {consecutive_failures} 次，关闭线程", console=True)
                break
            
            # 从队列获取任务
            try:
                task = task_queue.get(timeout=0.1)
                task_acquired = True
            except queue.Empty:
                break
            
            img_path = task['img_path']
            img_name = task['img_name']
            strat_name = task['strat_name']
            prompt_text = task['prompt_text']
            repeat_index = task['repeat_index']
            
            # 检查文件是否已存在
            file_prefix = f"{safe_file_name(img_path.stem)}_{strat_name}_{repeat_index}_gemini2.5-flash"
            existing_files = list(OUT_DIR.glob(f"{file_prefix}_*.txt"))
            
            if existing_files:
                log_and_print(f"[Thread-{thread_id}] 跳过: {img_name} | {strat_name} | 第{repeat_index}次 (已存在)")
                with stats_lock:
                    stats['skipped'] += 1
                pbar.update(1)
                task_queue.task_done()
                continue
            
            # 获取当前使用的模型
            current_model = MODELS[current_model_index]
            
            # 调用 API
            log_and_print(f"[Thread-{thread_id}|{current_model}] 开始: {img_name} | {strat_name} | 第{repeat_index}次")
            start_t = time.time()
            answer, should_switch_model, reasoning = get_response(
                client, prompt_text, img_path, current_model, thread_id
            )
            duration = time.time() - start_t
            
            # 检查是否是API错误
            if answer.startswith("API Error:"):
                # API 错误
                log_and_print(f"[Thread-{thread_id}] API错误，重新入队 ({duration:.2f}s): {answer[:200]}", level='warning')
                consecutive_failures += 1
                log_and_print(f"[Thread-{thread_id}] 连续失败: {consecutive_failures}/{THREAD_FAILURE_THRESHOLD}", console=True)
                
                with stats_lock:
                    stats['failed'] += 1
                
                task_queue.put(task)
                task_queue.task_done()
                
                if should_switch_model:
                    current_model_index = (current_model_index + 1) % len(MODELS)
                    log_and_print(f"[Thread-{thread_id}] 切换模型到: {MODELS[current_model_index]}", console=True)
                
                log_and_print(f"[Thread-{thread_id}] 等待 {FAILURE_SLEEP} 秒...")
                time.sleep(FAILURE_SLEEP)
                continue
            
            # API调用成功
            if consecutive_failures > 0:
                log_and_print(f"[Thread-{thread_id}] 任务成功，重置失败计数器 ({consecutive_failures} → 0)")
            consecutive_failures = 0
            
            # 生成文件名
            time_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            file_name = f"{file_prefix}_{time_str}.txt"
            out_path = OUT_DIR / file_name
            
            # 填充模板
            context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            full_content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer)
            
            # 将思考过程追加到文件末尾（与原脚本格式一致）
            if reasoning:
                # 清理 reasoning 中的重复 JSON 代码块
                cleaned_reasoning = clean_reasoning_content(reasoning)
                full_content += "\n\n=== Thinking Process ===\n" + cleaned_reasoning
            
            out_path.write_text(full_content, encoding='utf-8')
            
            log_and_print(f"[Thread-{thread_id}] ✅ 完成: {img_name} | {strat_name} | 第{repeat_index}次 ({duration:.2f}s)")
            
            with stats_lock:
                stats['completed'] += 1
            
            pbar.update(1)
            task_queue.task_done()
            
            # 成功后sleep
            sleep_time = random.uniform(SUCCESS_SLEEP_MIN, SUCCESS_SLEEP_MAX)
            log_and_print(f"[Thread-{thread_id}] 等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
            
        except Exception as e:
            log_and_print(f"[Thread-{thread_id}] 未捕获异常: {e}", level='error', console=True)
            consecutive_failures += 1
            
            with stats_lock:
                stats['failed'] += 1
            
            if task_acquired and task is not None:
                try:
                    task_queue.put(task)
                    log_and_print(f"[Thread-{thread_id}] 任务已重新入队")
                except Exception as put_error:
                    log_and_print(f"[Thread-{thread_id}] 重新入队失败: {put_error}", level='error')
                
                task_queue.task_done()
    
    # 线程结束 - 减少活跃线程计数
    with active_threads_lock:
        active_threads -= 1
        remaining = active_threads
    
    log_and_print(f"🏁 Thread-{thread_id} 结束，剩余 {remaining} 个线程在工作", console=True)


def main():
    """主函数"""
    global stop_flag
    
    # 设置优雅退出信号处理器
    def signal_handler(sig, frame):
        global stop_flag
        print("\n\n⚠️  接收到中断信号 (Ctrl+C)，正在优雅退出...")
        print("    等待当前任务完成后停止线程...")
        stop_flag = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if not PIC_DIR.exists():
        print(f"❌ 目录不存在：{PIC_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志系统
    log_file = setup_logging(OUT_DIR)
    
    images = [p for p in PIC_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    # 计算总任务数
    total_tasks = len(images) * len(STRATEGIES) * REPEAT_TIMES
    
    print(f"[*] 发现 {len(images)} 张图片，准备测试 {len(STRATEGIES)} 种策略。")
    print(f"[*] 每个策略重复 {REPEAT_TIMES} 次")
    print(f"[*] 模型策略: {MODELS}")
    print(f"[*] 并发线程数: {NUM_THREADS}")
    print(f"[*] 总计任务数: {total_tasks}")
    print(f"[*] Gemini Manager: {GEMINI_MANAGER_BASE_URL}")
    print(f"[*] 输出目录: {OUT_DIR}\n")
    
    # 创建任务队列
    task_queue = queue.Queue()
    
    print("正在构建任务队列...")
    for img in images:
        for strat_name, prompt_text in STRATEGIES.items():
            for i in range(REPEAT_TIMES):
                task_queue.put({
                    'img_path': img,
                    'img_name': img.name,
                    'strat_name': strat_name,
                    'prompt_text': prompt_text,
                    'repeat_index': i + 1
                })
    
    print(f"任务队列已构建，共 {task_queue.qsize()} 个任务\n")
    print(f"[*] 线程失败阈值: {THREAD_FAILURE_THRESHOLD} 次连续失败\n")
    
    # 创建进度条
    pbar = tqdm(total=total_tasks, unit="任务", desc="进度", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # 启动线程池
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i in range(NUM_THREADS):
            futures.append(executor.submit(process_with_thread, i, task_queue, pbar))
        
        # 等待所有线程完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"线程异常: {e}")
    
    pbar.close()
    
    # 检查剩余任务
    remaining_tasks = task_queue.qsize()
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"执行统计:")
    print(f"{'='*60}")
    print(f"✅ 成功完成: {stats['completed']} 个任务")
    print(f"⏭️  跳过（已存在）: {stats['skipped']} 个任务")
    print(f"❌ 失败（已重试）: {stats['failed']} 次")
    print(f"📋 剩余队列: {remaining_tasks} 个任务")
    print(f"{'='*60}\n")
    
    if remaining_tasks > 0:
        print(f"⚠️ 警告: 所有线程已结束，但队列中还有 {remaining_tasks} 个未完成的任务")
        print(f"   可能原因: 所有线程都已达到连续失败阈值")
        print(f"   建议: 检查网络连接、API 配额或稍后重新运行脚本")
    else:
        print(f"🎉 所有测试已完成！结果保存在 {OUT_DIR} 目录。")


if __name__ == "__main__":
    main()
