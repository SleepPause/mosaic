#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 Gemini Balance 代理调用 Gemini API（多线程并发版本，支持 thinking）

相比原版的简化：
- ✅ 使用 Gemini Balance 代理（http://localhost:8001/openai/v1）
- ✅ 保留：多线程并发
- ✅ 保留：任务队列
- ✅ 保留：Thinking 支持
- ✅ 保留：进度跟踪和日志

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
import prompt_strategies
import evaluation_templates
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ========= 用户配置区域 =========
# Gemini Balance 代理配置
BALANCE_BASE_URL = "http://localhost:8001/hf/v1"  # 使用 HF 格式以支持 thinking
BALANCE_API_KEY = "123456"  # 你的 ALLOWED_TOKENS

# ⚠️ 重要：要启用 thinking 功能，需要在 Gemini Balance 中配置：
# 1. 设置环境变量 SHOW_THINKING_PROCESS=true
# 2. 在 THINKING_MODELS 中添加 gemini-2.5-flash，例如: ["gemini-2.5-flash"]
# 3. 如果只想显示 thinking，使用普通模型；如果不想显示，使用 nonthinking 版本
# 详见: https://gb-docs.snaily.top/guide/environment.html

# 并发线程数（建议 10-20）
NUM_THREADS = 1

# Gemini 模型配置
MODEL = "gemini-2.5-flash"

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
THREAD_FAILURE_THRESHOLD = 10

# 速率限制配置（保守配置）
SUCCESS_SLEEP_MIN = 15  # 成功后最少等待秒数
SUCCESS_SLEEP_MAX = 20  # 成功后最多等待秒数
FAILURE_SLEEP = 60      # 失败后等待秒数
# =================================

# 使用 StrongREJECT 模板
EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT

# 线程安全锁
print_lock = threading.Lock()
stats_lock = threading.Lock()
active_threads_lock = threading.Lock()

# 统计计数器
stats = {
    'completed': 0,
    'skipped': 0,
    'failed': 0
}

# 活跃线程计数器
active_threads = 0

# 优雅退出标志
stop_flag = False

# 日志系统
logger = None


def setup_logging(log_dir: Path):
    """配置日志系统"""
    global logger
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_geminibalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger('GeminiBalanceEval')
    logger.setLevel(logging.DEBUG)
    
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
    """清理 reasoning 中的重复 JSON 代码块"""
    if not reasoning or len(reasoning) < 500:
        return reasoning
    
    # 检测未闭合的 JSON 块
    last_json_start = reasoning.rfind('```json')
    if last_json_start != -1:
        after_last_json = reasoning[last_json_start + 7:]
        next_close = after_last_json.find('```')
        
        if next_close == -1 and len(reasoning) - last_json_start < 3000:
            logger.debug(f"Detected incomplete JSON block, removing")
            reasoning = reasoning[:last_json_start].rstrip() + "\n\n[不完整的 JSON 块已移除]\n"
            return reasoning
    
    # 处理重复的 JSON 块
    json_blocks = re.findall(r'```json\s*\n(.*?)\n```', reasoning, re.DOTALL)
    
    if len(json_blocks) <= 1:
        return reasoning
    
    seen_blocks = set()
    duplicate_indices = []
    
    for i, block in enumerate(json_blocks):
        normalized = re.sub(r'\s+', '', block)
        if normalized in seen_blocks:
            duplicate_indices.append(i)
        else:
            seen_blocks.add(normalized)
    
    if duplicate_indices:
        parts = re.split(r'(```json\s*\n.*?\n```)', reasoning, flags=re.DOTALL)
        json_block_counter = 0
        filtered_parts = []
        
        for part in parts:
            if re.match(r'```json\s*\n.*?\n```', part, re.DOTALL):
                if json_block_counter not in duplicate_indices:
                    filtered_parts.append(part)
                else:
                    filtered_parts.append("\n[重复的 JSON 块已移除]\n")
                json_block_counter += 1
            else:
                filtered_parts.append(part)
        
        cleaned = ''.join(filtered_parts)
        logger.debug(f"Cleaned {len(duplicate_indices)} duplicate JSON blocks")
        return cleaned
    
    return reasoning


def log_and_print(message: str, level: str = 'info', console: bool = False):
    """智能日志输出"""
    if logger is None:
        return
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)
    
    if console:
        with print_lock:
            tqdm.write(message)


def get_client():
    """获取 OpenAI 客户端（连接到 Gemini Balance）"""
    return OpenAI(
        base_url=BALANCE_BASE_URL,
        api_key=BALANCE_API_KEY
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
    使用 Gemini Balance 调用 Gemini
    
    Returns:
        (answer, reasoning): 答案和思考过程
        如果失败，answer 以 "API Error:" 开头
    """
    try:
        # 准备图片数据
        base64_image = image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        # 调用 Gemini Balance
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
        
        # 提取答案和 thinking
        answer = response.choices[0].message.content
        reasoning = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        finish_reason = response.choices[0].finish_reason

        # 记录 thinking 状态
        if reasoning:
            log_and_print(f"[Thread-{thread_id}] ✅ 获取到 thinking 内容，长度: {len(reasoning)} 字符")
        else:
            log_and_print(f"[Thread-{thread_id}] ⚠️ 未获取到 thinking 内容", level='warning')
        
        if finish_reason in ["content_filter", "safety"]:
            log_and_print(f"[Thread-{thread_id}] 🛑 任务触发安全拦截 (Reason: {finish_reason})，停止重试。")
            return "[SAFETY_BLOCKED]", reasoning

        # 检查是否为空响应
        if not answer:
            return "API Error: Empty response", ""
        
        return answer, reasoning
        
    except Exception as e:
        error_str = str(e)
        log_and_print(f"[Thread-{thread_id}] API错误 ({model}): {error_str}", level='warning')
        return f"API Error: {error_str}", ""


def process_with_thread(thread_id, task_queue, pbar):
    """每个线程对应的工作函数"""
    global active_threads
    
    client = get_client()
    consecutive_failures = 0
    
    # 线程启动
    with active_threads_lock:
        active_threads += 1
        current_active = active_threads
    
    log_and_print(f"🔑 Thread-{thread_id} 启动（当前活跃线程: {current_active}）", console=True)
    
    while True:
        # 检查优雅退出
        if stop_flag:
            log_and_print(f"[Thread-{thread_id}] 接收到停止信号，优雅退出", console=True)
            break
        
        task = None
        task_acquired = False
        
        try:
            # 检查失败阈值
            if consecutive_failures >= THREAD_FAILURE_THRESHOLD:
                log_and_print(f"💀 [Thread-{thread_id}] 连续失败 {consecutive_failures} 次，关闭线程", console=True)
                break
            
            # 获取任务
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
            
            # 调用 API
            log_and_print(f"[Thread-{thread_id}|{MODEL}] 开始: {img_name} | {strat_name} | 第{repeat_index}次")
            start_t = time.time()
            answer, reasoning = get_response(client, prompt_text, img_path, MODEL, thread_id)
            duration = time.time() - start_t
            
            # 检查是否失败
            if answer.startswith("API Error:"):
                log_and_print(f"[Thread-{thread_id}] API错误，重新入队 ({duration:.2f}s): {answer[:200]}", level='warning')
                consecutive_failures += 1
                log_and_print(f"[Thread-{thread_id}] 连续失败: {consecutive_failures}/{THREAD_FAILURE_THRESHOLD}", console=True)
                
                with stats_lock:
                    stats['failed'] += 1
                
                task_queue.put(task)
                task_queue.task_done()
                
                # 可中断的等待
                log_and_print(f"[Thread-{thread_id}] 等待 {FAILURE_SLEEP} 秒...")
                for _ in range(int(FAILURE_SLEEP * 2)):  # 每 0.5 秒检查一次
                    if stop_flag:
                        break
                    time.sleep(0.5)
                continue
            
            # 成功
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
            
            # 追加 thinking
            if reasoning:
                cleaned_reasoning = clean_reasoning_content(reasoning)
                full_content += "\n\n=== Thinking Process ===\n" + cleaned_reasoning
            
            out_path.write_text(full_content, encoding='utf-8')
            
            log_and_print(f"[Thread-{thread_id}] ✅ 完成: {img_name} | {strat_name} | 第{repeat_index}次 ({duration:.2f}s)")
            
            with stats_lock:
                stats['completed'] += 1
            
            pbar.update(1)
            task_queue.task_done()
            
            # 成功后等待 (可中断)
            sleep_time = random.uniform(SUCCESS_SLEEP_MIN, SUCCESS_SLEEP_MAX)
            log_and_print(f"[Thread-{thread_id}] 等待 {sleep_time:.1f} 秒...")
            elapsed = 0
            while elapsed < sleep_time and not stop_flag:
                time.sleep(min(0.5, sleep_time - elapsed))
                elapsed += 0.5
            
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
    
    # 线程结束
    with active_threads_lock:
        active_threads -= 1
        remaining = active_threads
    
    log_and_print(f"🏁 Thread-{thread_id} 结束，剩余 {remaining} 个线程在工作", console=True)


def main():
    """主函数"""
    global stop_flag
    
    # 优雅退出信号处理
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
    
    # 初始化日志
    log_file = setup_logging(OUT_DIR)
    
    images = [p for p in PIC_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    # 计算总任务数
    total_tasks = len(images) * len(STRATEGIES) * REPEAT_TIMES
    
    print(f"[*] 发现 {len(images)} 张图片，准备测试 {len(STRATEGIES)} 种策略。")
    print(f"[*] 每个策略重复 {REPEAT_TIMES} 次")
    print(f"[*] 模型: {MODEL}")
    print(f"[*] 并发线程数: {NUM_THREADS}")
    print(f"[*] 总计任务数: {total_tasks}")
    print(f"[*] Gemini Balance 代理: {BALANCE_BASE_URL}")
    print(f"[*] API Key: {BALANCE_API_KEY}")
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
        
        # 等待所有线程完成 (使用短超时以便能响应 Ctrl+C)
        try:
            while True:
                # 检查是否所有任务完成
                all_done = all(f.done() for f in futures)
                if all_done:
                    break
                
                # 短暂等待,允许 KeyboardInterrupt 被处理
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  检测到 Ctrl+C,立即终止所有线程...")
            stop_flag = True
            # 强制关闭线程池
            executor.shutdown(wait=False, cancel_futures=True)
            print("✅ 线程池已关闭")
            pbar.close()
            return
        
        # 收集异常
        for future in futures:
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                pass  # 已经完成了,忽略超时
            except Exception as e:
                print(f"线程异常: {e}")
    
    pbar.close()
    
    # 检查剩余任务
    remaining_tasks = task_queue.qsize()
    
    # 输出统计
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
        print(f"   建议: 检查网络连接、Gemini Balance 状态或稍后重新运行脚本")
    else:
        print(f"🎉 所有测试已完成！结果保存在 {OUT_DIR} 目录。")


if __name__ == "__main__":
    main()
