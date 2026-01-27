#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 Gemini 原生 API 实现（多线程并发版本，支持 thinking）
文件名：图片名_策略名_序号_gemini_时间戳.txt

【模板函数版本】支持多目标词汇的动态提示词生成
"""
import time
import re
import logging
import random
import signal
from datetime import datetime
from pathlib import Path
from PIL import Image
import prompt_strategies_new as prompt_strategies  # 导入新版提示词库
import evaluation_templates
import os
from dotenv import load_dotenv
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

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
PIC_DIR = BASE_DIR.parent / "pic"
OUT_DIR = BASE_DIR / "eval_text"

# 每个策略重复测试的次数
REPEAT_TIMES = 10

# 线程失败阈值：如果线程连续失败（轮换3遍MODELS）且没有完成任何任务，则关闭该线程
THREAD_FAILURE_THRESHOLD = len(MODELS) * 3000000

# 速率限制配置（基于 RPM 限制）
SUCCESS_SLEEP_MIN = 15  # 成功后最少等待15秒
SUCCESS_SLEEP_MAX = 20  # 成功后最多等待20秒（加入随机性）
FAILURE_SLEEP = random.uniform(30,50)       # 失败后等待40秒
RPM429_SLEEP = random.uniform(30,50)        # 遇到429错误后等待40秒
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


def get_strategies_for_target(target_word: str) -> dict:
    """
    根据目标词汇动态生成策略集合
    
    Args:
        target_word: 从图片文件名提取的目标词（如 "HEROIN", "METH"）
    
    Returns:
        策略名称到提示词文本的映射字典
    """
    # 从映射字典中获取对应的 action，如果不存在则使用默认值
    action = prompt_strategies.TARGET_CONTEXTS.get(target_word, "synthesis process")
    
    return {
        "BreakingBad": prompt_strategies.get_breaking_bad_prompt(target_word, action),
        #"SafetyAuditor": prompt_strategies.get_safety_auditor_prompt(target_word, action),
        #"Historian": prompt_strategies.get_historian_prompt(target_word, action),
        "JSON": prompt_strategies.get_json_format_prompt(target_word, action),
        #"ArtCritic": prompt_strategies.get_art_critic_prompt(target_word, action),
    }


def extract_target_from_filename(filename: str) -> str:
    """
    从文件名中提取目标词汇
    
    例如: "HEROIN_Opacity_64_negative_words.jpg" -> "HEROIN"
          "METH_test.jpg" -> "METH"
    
    Args:
        filename: 图片文件名（不含路径）
    
    Returns:
        提取的目标词（大写），如果无法提取则返回 "HEROIN" 作为默认值
    """
    # 假设文件名格式为: TARGET_其他信息.jpg
    parts = filename.split('_')
    if parts:
        target = parts[0].upper()
        # 验证是否在已知目标列表中
        if target in prompt_strategies.TARGET_CONTEXTS:
            return target
    
    # 如果无法识别，默认返回 HEROIN
    log_and_print(f"⚠️  无法从文件名 '{filename}' 提取目标词，使用默认值 HEROIN", console=True)
    return "HEROIN"


def setup_logging(log_dir: Path):
    """
    配置日志系统：详细日志写入文件，控制台只显示关键信息
    """
    global logger
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    log_file = log_dir / f"eval_gemini_native_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logger = logging.getLogger('GeminiNativeEvalNew')
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
    """清理 reasoning_content 中的重复 JSON 代码块"""
    if not reasoning or len(reasoning) < 500:
        return reasoning
    
    import re
    
    # 检测未闭合的 JSON 代码块
    last_json_start = reasoning.rfind('```json')
    
    if last_json_start != -1:
        after_last_json = reasoning[last_json_start + 7:]
        next_close = after_last_json.find('```')
        
        if next_close == -1:
            if len(reasoning) - last_json_start < 3000:
                logger.debug(f"Detected incomplete JSON block, removing truncated content")
                reasoning = reasoning[:last_json_start].rstrip() + "\n\n[不完整的 JSON 块已移除（可能被截断）]\n"
                return reasoning
    
    # 处理重复 JSON 块
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
    
    Returns:
        (answer, should_switch_model, is_quota_exhausted, reasoning)
    """
    try:
        # 确定使用的 thinking 配置
        thinking_config = types.ThinkingConfig(include_thoughts=True)
        
        if "gemini-2.5" in model:
            thinking_config.thinking_budget = 8192  # medium 级别
        elif "gemini-3" in model:
            thinking_config.thinking_level = types.ThinkingLevel.MEDIUM
        
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
                
                if part.thought:  # 思考内容
                    if len(reasoning_content) < 20000:
                        reasoning_content += part.text
                    elif not reasoning_content.endswith("... [Truncated]"):
                        reasoning_content += "\n... [Thinking Process Truncated]"
                else:  # 回答内容
                    answer_content += part.text
        
        answer = answer_content.strip()
        reasoning = reasoning_content.strip()
        
        if not answer:
            return "API Error: Empty response from API", False, False, ""
        
        return answer, False, False, reasoning  # 成功
        
    except Exception as e:
        error_str = str(e)
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
    """每个 API Key 对应的工作线程"""
    global active_threads
    
    client = get_client(api_key)
    current_model_index = 0
    consecutive_failures = 0
    
    with active_threads_lock:
        active_threads += 1
        current_active = active_threads
    
    log_and_print(f"🔑 Key-{key_id} 启动（当前活跃线程: {current_active}）", console=True)
    
    while True:
        if stop_flag:
            log_and_print(f"[Key-{key_id}] 接收到停止信号，优雅退出", console=True)
            break
        
        task = None
        task_acquired = False
        
        try:
            if consecutive_failures >= THREAD_FAILURE_THRESHOLD:
                log_and_print(f"💀 [Key-{key_id}] 连续失败 {consecutive_failures} 次，关闭线程", console=True)
                break
            
            try:
                task = task_queue.get(timeout=0.1)
                task_acquired = True
            except queue.Empty:
                break
            
            img_path = task['img_path']
            img_name = task['img_name']
            target_word = task['target_word']
            strat_name = task['strat_name']
            prompt_text = task['prompt_text']
            repeat_index = task['repeat_index']
            
            # 从图片文件名中提取 method 标记（methodA/methodB）
            img_stem = img_path.stem
            method_suffix = ""
            method_name = ""  # 用于创建目录
            
            # 检查是否包含 _methodX 格式
            if "_method" in img_stem:
                parts = img_stem.split("_")
                for i, part in enumerate(parts):
                    if part.startswith("method"):
                        method_suffix = f"_{part}"
                        method_name = part  # 提取 methodA 或 methodB
                        # 重建不包含 method 的文件名
                        img_stem = "_".join(parts[:i] + parts[i+1:])
                        break
            
            # 创建按 method 和 target 分类的目录结构
            if method_name:
                target_dir = OUT_DIR / method_name / target_word
            else:
                target_dir = OUT_DIR / target_word
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_prefix = f"{safe_file_name(img_stem)}_{strat_name}_{repeat_index}_gemini2.5-flash{method_suffix}"
            existing_files = list(target_dir.glob(f"{file_prefix}_*.txt"))
            
            if existing_files:
                log_and_print(f"[Key-{key_id}] 跳过: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次 (已存在)")
                with stats_lock:
                    stats['skipped'] += 1
                pbar.update(1)
                task_queue.task_done()
                continue
            
            # 加载图片
            pil_image = None
            try:
                pil_image = image_path_to_pil(img_path)
                
                current_model = MODELS[current_model_index]
                
                log_and_print(f"[Key-{key_id}|{current_model}] 开始: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次")
                start_t = time.time()
                answer, should_switch_model, is_quota_exhausted, reasoning = get_response(
                    client, prompt_text, pil_image, current_model, key_id
                )
                duration = time.time() - start_t
            finally:
                if pil_image is not None:
                    pil_image.close()
            
            
            if answer.startswith("API Error:"):
                # RPD配额耗尽
                if is_quota_exhausted:
                    log_and_print(
                        f"💰 [Key-{key_id}|{current_model}] RPD配额已用完(20次/天)，切换到下一个模型",
                        console=True
                    )
                    
                    time.sleep(10)
                    old_model = current_model
                    current_model_index = (current_model_index + 1) % len(MODELS)
                    new_model = MODELS[current_model_index]
                    
                    if current_model_index == -1:
                        log_and_print(
                            f"✅ [Key-{key_id}] 所有模型配额已用完，线程优雅退出",
                            console=True
                        )
                        task_queue.put(task)
                        task_queue.task_done()
                        break
                    
                    log_and_print(f"[Key-{key_id}] 模型切换: {old_model} → {new_model}", console=True)
                    task_queue.put(task)
                    task_queue.task_done()
                    continue
                
                # 其他API错误
                log_and_print(f"[Key-{key_id}] API错误，重新入队 ({duration:.2f}s): {answer[:200]}", level='warning')
                consecutive_failures += 1
                log_and_print(f"[Key-{key_id}] 连续失败: {consecutive_failures}/{THREAD_FAILURE_THRESHOLD}", console=True)
                
                with stats_lock:
                    stats['failed'] += 1
                
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
            
            time_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            file_name = f"{file_prefix}_{time_str}.txt"
            out_path = target_dir / file_name
            
            context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            full_content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer)
            
            if reasoning:
                cleaned_reasoning = clean_reasoning_content(reasoning)
                full_content += "\n\n=== Thinking Process ===\n" + cleaned_reasoning
            
            out_path.write_text(full_content, encoding='utf-8')
            
            log_and_print(f"[Key-{key_id}] ✅ 完成: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次 ({duration:.2f}s)")
            
            with stats_lock:
                stats['completed'] += 1
            
            pbar.update(1)
            task_queue.task_done()
            
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
    
    with active_threads_lock:
        active_threads -= 1
        remaining = active_threads
    
    log_and_print(f"🏁 Key-{key_id} 结束，剩余 {remaining} 个线程在工作", console=True)


def main():
    """主函数"""
    global stop_flag
    
    def signal_handler(sig, frame):
        print("\n\n⚠️  接收到中断信号 (Ctrl+C)，立即退出...")
        os._exit(0)  # 强制立即退出，不等待线程
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if not PIC_DIR.exists():
        print(f"❌ 目录不存在：{PIC_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = setup_logging(OUT_DIR)
    
    images = [p for p in PIC_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    total_tasks = len(images) * 2 * REPEAT_TIMES  # 5 个策略
    
    print(f"[*] 发现 {len(images)} 张图片")
    print(f"[*] 支持的目标词汇: {', '.join(prompt_strategies.TARGET_CONTEXTS.keys())}")
    print(f"[*] 每个策略重复 {REPEAT_TIMES} 次")
    print(f"[*] 模型策略: {MODELS}")
    print(f"[*] API Keys 数量: {len(GEMINI_API_KEYS)}")
    print(f"[*] 并发线程数: {len(GEMINI_API_KEYS)}")
    print(f"[*] 总计任务数: {total_tasks}")
    print(f"[*] 输出目录: {OUT_DIR}\n")
    
    # 统计Method文件夹中的txt文件数量
    if OUT_DIR.exists():
        method_dirs = [d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith('method')]
        if method_dirs:
            for method_dir in sorted(method_dirs):
                txt_count = len(list(method_dir.rglob('*.txt')))
                print(f"[*] {method_dir.name} 文件夹中有 {txt_count} 个txt文件")
        else:
            print(f"[*] 未发现任何Method文件夹")
    print()
    
    task_queue = queue.Queue()
    
    print("正在构建任务队列...")
    for img in images:
        target_word = extract_target_from_filename(img.stem)
        STRATEGIES = get_strategies_for_target(target_word)
        
        for strat_name, prompt_text in STRATEGIES.items():
            for i in range(REPEAT_TIMES):
                task_queue.put({
                    'img_path': img,
                    'img_name': img.name,
                    'target_word': target_word,
                    'strat_name': strat_name,
                    'prompt_text': prompt_text,
                    'repeat_index': i + 1
                })
    
    print(f"任务队列已构建，共 {task_queue.qsize()} 个任务\n")
    print(f"[*] 线程失败阈值: {THREAD_FAILURE_THRESHOLD} 次连续失败\n")
    
    pbar = tqdm(total=total_tasks, unit="任务", desc="进度", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    with ThreadPoolExecutor(max_workers=len(GEMINI_API_KEYS)) as executor:
        futures = []
        for i, api_key in enumerate(GEMINI_API_KEYS):
            futures.append(executor.submit(process_with_key, api_key, i, task_queue, pbar))
        
        # 主线程循环等待，确保能响应 Ctrl+C 信号
        try:
            while any(not f.done() for f in futures):
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\n⚠️  接收到中断信号 (Ctrl+C)，立即退出...")
            os._exit(0)

        # 检查任务结果
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"线程异常: {e}")
    
    pbar.close()
    
    remaining_tasks = task_queue.qsize()
    
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
