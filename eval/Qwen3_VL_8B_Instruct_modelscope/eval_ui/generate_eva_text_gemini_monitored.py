#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 Gemini 原生 API 实现（多线程并发版本，支持 thinking，带监控）
文件名：图片名_策略名_序号_gemini_时间戳.txt

注意：此版本集成了监控服务，需配合 ws_server.py 使用
"""
import time
import re
import logging
import random
import signal
from datetime import datetime
from pathlib import Path
from PIL import Image
import prompt_strategies
import evaluation_templates
import os
from dotenv import load_dotenv
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

# 导入监控服务
from monitor_service import monitor

load_dotenv()

# ========= 用户配置区域 =========
# 获取 Gemini API Keys
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("未找到 GEMINI_API_KEYS 环境变量")
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(",") if key.strip()]

# Gemini 模型配置（循环队列）
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

# 使用脚本所在目录作为基准，防止在不同目录下运行时找不到文件
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"  # 注意：eval_ui 是子文件夹，需要向上两级
OUT_DIR = BASE_DIR.parent / "eval" / "eval_text"  # 输出到原位置

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

# 线程失败阈值：如果线程连续失败（轮换3遍MODELS）且没有完成任何任务，则关闭该线程
THREAD_FAILURE_THRESHOLD = len(MODELS) * 3

# 速率限制配置（基于 RPM 限制）
SUCCESS_SLEEP_MIN = 15  # 成功后最少等待30秒
SUCCESS_SLEEP_MAX = 20  # 成功后最多等待40秒（加入随机性）
FAILURE_SLEEP = 40       # 失败后等待90秒
RPM429_SLEEP = 40       # 遇到429错误后等待120秒
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
    log_file = log_dir / f"eval_gemini_native_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logger = logging.getLogger('GeminiNativeEval')
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
            # 检查这个未闭合的块是否在文件末尾附近（最后5000字符内）
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


def get_client(api_key):
    """获取 Gemini 原生客户端"""
    return genai.Client(api_key=api_key)


def image_path_to_pil(image_path: Path) -> Image.Image:
    """将图片路径转换为 PIL Image 对象"""
    return Image.open(image_path)


def safe_file_name(name: str) -> str:
    """去掉特殊符号"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_response(client, text_prompt: str, pil_image: Image.Image, model: str, key_id: int) -> tuple:
    """
    调用 Gemini 原生 API 获取回答（流式响应，支持 thinking）
    只尝试一次，不做内部重试
    
    Returns:
        (answer, should_switch_model, is_quota_exhausted, reasoning): 
            answer - API响应或错误信息
            should_switch_model - 是否应该切换模型
            is_quota_exhausted - 是否是RPD配额耗尽
            reasoning - 思考过程（thinking process）
    """
    try:
        # 确定使用的 thinking 配置
        # Gemini 2.5: 使用 thinking_budget
        # Gemini 3: 使用 thinking_level
        thinking_config = types.ThinkingConfig(include_thoughts=True)
        
        if "gemini-2.5" in model:
            thinking_config.thinking_budget = 8192  # medium 级别
        elif "gemini-3" in model:
            thinking_config.thinking_level = types.ThinkingLevel.MEDIUM
        # else: 不设置具体参数，使用默认值
        
        # 调用 Gemini 原生 API（流式）
        response = client.models.generate_content_stream(
            model=model,
            contents=[text_prompt, pil_image],
            config=types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=thinking_config
            )
        )
        
        answer_content = ""
        reasoning_content = ""
        
        # 处理流式响应
        for chunk in response:
            if not chunk.candidates:
                continue
            
            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                
                # 判断是思考内容还是回答内容
                if part.thought:  # 思考内容
                    if len(reasoning_content) < 20000:
                        reasoning_content += part.text
                    elif not reasoning_content.endswith("... [Truncated]"):
                        reasoning_content += "\n... [Thinking Process Truncated]"
                else:  # 回答内容
                    answer_content += part.text
        
        answer = answer_content.strip()
        reasoning = reasoning_content.strip()
        
        # 检查响应是否为空
        if not answer:
            return "API Error: Empty response from API", False, False, ""
        
        return answer, False, False, reasoning  # 成功
        
    except Exception as e:
        error_str = str(e)
        # 记录到日志文件
        log_and_print(f"[Key-{key_id}] API错误 ({model}): {error_str}", level='warning')
        
        # 识别RPD配额耗尽
        is_quota_exhausted = any(phrase in error_str for phrase in [
            "exceeded your current quota",
            "Quota exceeded for quota metric",
            "quota_limit_per_day"
        ])
        
        # 检查是否是需要切换模型的错误
        should_switch = any(code in error_str for code in [
            "429", "RESOURCE_EXHAUSTED", "503", "500", "403", 
            "400", "404", "504", "Quota", "timeout", "timed out", "Timeout"
        ]) and not is_quota_exhausted
        
        return f"API Error: {error_str}", should_switch, is_quota_exhausted, ""


def process_with_key(api_key, key_id, task_queue, pbar):
    """
    每个 API Key 对应的工作线程
    """
    global active_threads
    
    client = get_client(api_key)
    current_model_index = 0
    consecutive_failures = 0
    thread_stopped = False  # 标记线程是否已停止（防止重复调用 monitor）
    
    # 线程启动 - 增加活跃线程计数
    with active_threads_lock:
        active_threads += 1
        current_active = active_threads
    
    log_and_print(f"🔑 Key-{key_id} 启动（当前活跃线程: {current_active}）", console=True)
    
    # 注册线程到监控服务（在 main 函数中统一注册，这里不重复）
    
    while True:
        # 检查是否需要优雅退出
        if stop_flag:
            log_and_print(f"[Key-{key_id}] 接收到停止信号，优雅退出", console=True)
            monitor.mark_thread_stopped(key_id, '用户中断')
            break
        
        task = None
        task_acquired = False
        
        try:
            # 检查是否达到失败阈值
            if consecutive_failures >= THREAD_FAILURE_THRESHOLD:
                log_and_print(f"💀 [Key-{key_id}] 连续失败 {consecutive_failures} 次，关闭线程", console=True)
                monitor.mark_thread_stopped(key_id, f'连续失败{consecutive_failures}次')
                break
            
            # 从队列获取任务
            try:
                task = task_queue.get(timeout=0.1)
                task_acquired = True
            except queue.Empty:
                monitor.mark_thread_stopped(key_id, '队列已空')
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
                # 跳过文件只输出到控制台，不写入日志（避免断点续传时日志过大）
                # log_and_print(f"[Key-{key_id}] 跳过: {img_name} | {strat_name} | 第{repeat_index}次 (已存在)")
                with stats_lock:
                    stats['skipped'] += 1
                pbar.update(1)
                # 更新监控：跳过任务
                monitor.update_task_complete(key_id, success=False, skipped=True)
                task_queue.task_done()
                continue
            
            # 加载图片（使用 try-finally 确保关闭）
            pil_image = None
            try:
                pil_image = image_path_to_pil(img_path)
                
                # 获取当前使用的模型
                current_model = MODELS[current_model_index]
                
                # 更新监控：任务开始
                monitor.update_task_start(key_id, img_name, strat_name, repeat_index)
                
                # 调用 API
                log_and_print(f"[Key-{key_id}|{current_model}] 开始: {img_name} | {strat_name} | 第{repeat_index}次")
                start_t = time.time()
                answer, should_switch_model, is_quota_exhausted, reasoning = get_response(
                    client, prompt_text, pil_image, current_model, key_id
                )
                duration = time.time() - start_t
            finally:
                # 确保图片资源被释放
                if pil_image is not None:
                    pil_image.close()
            
            
            # 检查是否是API错误
            if answer.startswith("API Error:"):
                # RPD配额耗尽
                if is_quota_exhausted:
                    log_and_print(
                        f"💰 [Key-{key_id}|{current_model}] RPD配额已用完(20次/天)，切换到下一个模型",
                        console=True
                    )
                    
                    old_model = current_model
                    current_model_index = (current_model_index + 1) % len(MODELS)
                    new_model = MODELS[current_model_index]
                    
                    if current_model_index == 0:
                        log_and_print(
                            f"✅ [Key-{key_id}] 所有模型配额已用完，线程优雅退出",
                            console=True
                        )
                        monitor.mark_thread_stopped(key_id, '所有模型配额用尽')
                        thread_stopped = True
                        task_queue.put(task)
                        task_queue.task_done()
                        break
                    
                    log_and_print(f"[Key-{key_id}] 模型切换: {old_model} → {new_model}", console=True)
                    # 更新监控：切换模型
                    monitor.update_model(key_id, new_model)
                    task_queue.put(task)
                    task_queue.task_done()
                    continue
                
                # 其他API错误
                log_and_print(f"[Key-{key_id}] API错误，重新入队 ({duration:.2f}s): {answer[:200]}", level='warning')
                consecutive_failures += 1
                log_and_print(f"[Key-{key_id}] 连续失败: {consecutive_failures}/{THREAD_FAILURE_THRESHOLD}", console=True)
                
                with stats_lock:
                    stats['failed'] += 1
                
                # 更新监控：任务失败
                monitor.update_task_complete(key_id, success=False, skipped=False)
                
                task_queue.put(task)
                task_queue.task_done()
                
                if should_switch_model:
                    current_model_index = (current_model_index + 1) % len(MODELS)
                    log_and_print(f"[Key-{key_id}] 切换模型到: {MODELS[current_model_index]}", console=True)
                
                is_429_error = "429" in answer or "RESOURCE_EXHAUSTED" in answer
                sleep_time = RPM429_SLEEP if is_429_error else FAILURE_SLEEP
                log_and_print(f"[Key-{key_id}] 等待 {sleep_time} 秒...")
                time.sleep(sleep_time)
                continue
            
            # API调用成功
            if consecutive_failures > 0:
                log_and_print(f"[Key-{key_id}] 任务成功，重置失败计数器 ({consecutive_failures} → 0)")
            consecutive_failures = 0
            
            # 生成文件名
            time_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            file_name = f"{file_prefix}_{time_str}.txt"
            out_path = OUT_DIR / file_name
            
            # 填充模板
            context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            full_content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer)
            
            # 将思考过程追加到文件末尾（与 Qwen 脚本格式完全一致）
            if reasoning:
                # 清理 reasoning 中的重复 JSON 代码块（特别针对 JSON 策略）
                cleaned_reasoning = clean_reasoning_content(reasoning)
                full_content += "\n\n=== Thinking Process ===\n" + cleaned_reasoning
            
            out_path.write_text(full_content, encoding='utf-8')
            
            log_and_print(f"[Key-{key_id}] ✅ 完成: {img_name} | {strat_name} | 第{repeat_index}次 ({duration:.2f}s)")
            
            with stats_lock:
                stats['completed'] += 1
            
            # 更新监控：任务成功
            monitor.update_task_complete(key_id, success=True, skipped=False)
            
            pbar.update(1)
            task_queue.task_done()
            
            # 成功后sleep
            sleep_time = random.uniform(SUCCESS_SLEEP_MIN, SUCCESS_SLEEP_MAX)
            log_and_print(f"[Key-{key_id}] 等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
            
        except Exception as e:
            log_and_print(f"[Key-{key_id}] 未捕获异常: {e}", level='error', console=True)
            consecutive_failures += 1
            
            with stats_lock:
                stats['failed'] += 1
            
            if task_acquired and task is not None:
                try:
                    task_queue.put(task)
                    log_and_print(f"[Key-{key_id}] 任务已重新入队")
                except Exception as put_error:
                    log_and_print(f"[Key-{key_id}] 重新入队失败: {put_error}", level='error')
                
                task_queue.task_done()
    
    # 线程结束 - 减少活跃线程计数
    with active_threads_lock:
        active_threads -= 1
        remaining = active_threads
    
    # 标记线程已停止（避免重复调用）
    if not thread_stopped:
        monitor.mark_thread_stopped(key_id)
    
    log_and_print(f"🏁 Key-{key_id} 结束，剩余 {remaining} 个线程在工作", console=True)


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
    print(f"[*] API Keys 数量: {len(GEMINI_API_KEYS)}")
    print(f"[*] 并发线程数: {len(GEMINI_API_KEYS)}")
    print(f"[*] 总计任务数: {total_tasks}")
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
    
    # 初始化监控服务
    print(f"[*] 监控面板: http://localhost:9000\n")
    monitor.reset()
    for i in range(len(GEMINI_API_KEYS)):
        monitor.register_thread(i, MODELS[0], len(GEMINI_API_KEYS), total_tasks)
    monitor.update_queue_size(task_queue.qsize())
    
    # 创建进度条
    pbar = tqdm(total=total_tasks, unit="任务", desc="进度", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # 启动线程池
    with ThreadPoolExecutor(max_workers=len(GEMINI_API_KEYS)) as executor:
        futures = []
        for i, api_key in enumerate(GEMINI_API_KEYS):
            futures.append(executor.submit(process_with_key, api_key, i, task_queue, pbar))
        
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
        print(f"   可能原因: 所有 API Keys 都已达到连续失败阈值")
        print(f"   建议: 检查网络连接、API 配额或稍后重新运行脚本")
    else:
        print(f"🎉 所有测试已完成！结果保存在 {OUT_DIR} 目录。")


if __name__ == "__main__":
    main()
