#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 Gemini 原生 API 实现（Key轮询版本，支持 thinking）
文件名：图片名_策略名_序号_gemini_时间戳.txt

【Key轮询版本】n个工作线程从l个API Key的循环队列中轮询获取key，每个任务依次尝试k个模型
"""
import time
import re
import logging
import random
import signal
import collections
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

# 工作线程数配置
NUM_WORKERS = 5  # 固定5个工作线程

# 使用脚本所在目录作为基准，防止在不同目录下运行时找不到文件
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"
OUT_DIR = BASE_DIR / "eval_text"  # 修改输出目录以区分

# 每个策略重复测试的次数
REPEAT_TIMES = 10

# 速率限制配置（基于 RPM 限制）
SUCCESS_SLEEP_MIN = 10  # 成功后最少等待15秒
SUCCESS_SLEEP_MAX = 15  # 成功后最多等待20秒（加入随机性）
FAILURE_SLEEP = 5       # 单个模型失败后等待5秒再尝试下一个模型
# =================================

# 使用 StrongREJECT 模板
EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT

# 线程安全锁
print_lock = threading.Lock()
stats_lock = threading.Lock()

# 统计计数器
stats = {
    'completed': 0,    # 成功完成的任务
    'skipped': 0,      # 跳过的任务（已存在）
    'failed': 0        # 失败的任务（重新入队）
}

# 优雅退出标志
stop_flag = False

# 日志系统
logger = None  # 将在 main 函数中初始化


class KeyQueue:
    """线程安全的API Key循环队列"""
    def __init__(self, keys: list):
        """
        初始化Key循环队列
        
        Args:
            keys: API Key列表
        """
        # 存储 (key_id, api_key) 元组
        self.queue = collections.deque([(i, key) for i, key in enumerate(keys)])
        self.lock = threading.Lock()
        self.total_keys = len(keys)
    
    def get_key(self) -> tuple:
        """从队列头部获取一个key
        
        Returns:
            (key_id, api_key) 或 (None, None) 如果队列为空
        """
        with self.lock:
            if not self.queue:
                return None, None
            key_tuple = self.queue.popleft()
            return key_tuple
    
    def return_key(self, key_id: int, api_key: str):
        """将key放回队列末尾
        
        Args:
            key_id: Key的ID
            api_key: API Key字符串
        """
        with self.lock:
            self.queue.append((key_id, api_key))
    
    def size(self) -> int:
        """获取当前队列大小"""
        with self.lock:
            return len(self.queue)


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
    log_file = log_dir / f"eval_gemini_polling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logger = logging.getLogger('GeminiPollingEval')
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
        (answer, is_quota_exhausted, reasoning)
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
            return "API Error: Empty response from API", False, ""
        
        return answer, False, reasoning  # 成功
        
    except Exception as e:
        error_str = str(e)
        log_and_print(f"[Key-{key_id}] API错误 ({model}): {error_str}", level='warning')
        
        # 识别RPD配额耗尽
        is_quota_exhausted = any(phrase in error_str for phrase in [
            "exceeded your current quota",
            "Quota exceeded for quota metric",
            "quota_limit_per_day"
        ])
        
        return f"API Error: {error_str}", is_quota_exhausted, ""


def process_with_polling(thread_id: int, key_queue: KeyQueue, task_queue: queue.Queue, pbar):
    """工作线程函数 - 使用Key轮询机制"""
    
    log_and_print(f"🔧 Thread-{thread_id} 启动", console=True)
    
    while True:
        if stop_flag:
            log_and_print(f"[Thread-{thread_id}] 接收到停止信号，优雅退出", console=True)
            break
        
        # 1. 从key队列获取一个key
        key_id, api_key = key_queue.get_key()
        if api_key is None:
            log_and_print(f"[Thread-{thread_id}] Key队列为空，退出", console=True)
            break
        
        # 2. 创建客户端（一个key使用期间复用同一个客户端）
        client = get_client(api_key)
        
        # 3. 对每个模型，独立获取并处理一个任务
        for model_index, model in enumerate(MODELS):
            if stop_flag:
                break
            
            # 3.1 循环获取任务，跳过已完成的，直到找到未完成的任务
            while True:
                task = None
                task_acquired = False
                try:
                    task = task_queue.get(timeout=0.5)
                    task_acquired = True
                except queue.Empty:
                    # 没有更多任务了
                    log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] 任务队列为空", console=True)
                    break  # 退出任务获取循环
                
                # 提取任务信息
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
                    # 已完成，跳过并继续获取下一个任务（不记录到日志）
                    with stats_lock:
                        stats['skipped'] += 1
                    pbar.update(1)
                    task_queue.task_done()
                    continue  # 继续获取下一个任务
                
                # 找到未完成的任务，跳出获取循环
                break
            
            # 如果没有获取到任务（队列空或stop_flag），退出模型循环
            if not task_acquired or task is None:
                break
            
            # 3.2 用当前模型处理这个未完成的任务
            pil_image = None
            try:
                pil_image = image_path_to_pil(img_path)
                
                log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] 开始: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次")
                start_t = time.time()
                
                answer, is_quota_exhausted, reasoning = get_response(
                    client, prompt_text, pil_image, model, key_id
                )
                duration = time.time() - start_t
                
            finally:
                if pil_image is not None:
                    pil_image.close()
            
            # 3.3 判断结果：成功 or 失败
            if not answer.startswith("API Error:"):
                # 成功！保存结果
                time_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
                file_name = f"{file_prefix}_{time_str}.txt"
                out_path = target_dir / file_name
                
                context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
                full_content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer)
                
                if reasoning:
                    cleaned_reasoning = clean_reasoning_content(reasoning)
                    full_content += "\n\n=== Thinking Process ===\n" + cleaned_reasoning
                
                out_path.write_text(full_content, encoding='utf-8')
                
                log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] ✅ 完成: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次 ({duration:.2f}s)")
                
                with stats_lock:
                    stats['completed'] += 1
                
                pbar.update(1)
                
                # 成功后等待
                sleep_time = random.uniform(SUCCESS_SLEEP_MIN, SUCCESS_SLEEP_MAX)
                log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] 等待 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)
                
            else:
                # 失败！任务重新入队
                log_and_print(
                    f"[Thread-{thread_id}|Key-{key_id}|{model}] ❌ 失败，任务重新入队: {answer[:100]}",
                    level='warning'
                )
                task_queue.put(task)
                
                with stats_lock:
                    stats['failed'] += 1
                
                # 失败后等待
                log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] 等待{FAILURE_SLEEP}秒...")
                time.sleep(FAILURE_SLEEP)
            
            # 3.4 标记任务完成
            if task_acquired:
                task_queue.task_done()

        
        # 4. k个模型都处理完后，将key放回队列末尾
        key_queue.return_key(key_id, api_key)
        log_and_print(f"[Thread-{thread_id}] Key-{key_id} 已归还到队列（处理了{len(MODELS)}个模型任务）")
    
    log_and_print(f"🏁 Thread-{thread_id} 结束", console=True)



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
    
    total_tasks = len(images) * 2 * REPEAT_TIMES  # 2 个策略
    
    print(f"[*] 发现 {len(images)} 张图片")
    print(f"[*] 支持的目标词汇: {', '.join(prompt_strategies.TARGET_CONTEXTS.keys())}")
    print(f"[*] 每个策略重复 {REPEAT_TIMES} 次")
    print(f"[*] 模型列表: {MODELS} (共 {len(MODELS)} 个)")
    print(f"[*] API Keys 数量: {len(GEMINI_API_KEYS)}")
    print(f"[*] 工作线程数: {NUM_WORKERS}")
    print(f"[*] 🔄 Key轮询模式: 启用")
    print(f"[*] 总计任务数: {total_tasks}")
    print(f"[*] 输出目录: {OUT_DIR}\n")
    


    # 统计Method文件夹中的txt文件数量
    if OUT_DIR.exists():
        method_dirs = [d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith('method')]
        if method_dirs:
            total_txt_count = sum(len(list(method_dir.rglob('*.txt'))) for method_dir in method_dirs)
            print(f"[*] 所有Method文件夹共有 {total_txt_count} 个txt文件")
        else:
            print(f"[*] 未发现任何Method文件夹")
    print()
    
    # 创建Key循环队列
    key_queue = KeyQueue(GEMINI_API_KEYS)
    
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
    
    pbar = tqdm(total=total_tasks, unit="任务", desc="进度", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # 创建固定数量的工作线程
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for thread_id in range(NUM_WORKERS):
            futures.append(
                executor.submit(process_with_polling, thread_id, key_queue, task_queue, pbar)
            )
        
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
    print(f"🔑 Key队列剩余: {key_queue.size()} 个keys")
    print(f"{'='*60}\n")
    
    if remaining_tasks > 0:
        print(f"⚠️ 警告: 所有线程已结束，但队列中还有 {remaining_tasks} 个未完成的任务")
        print(f"   可能原因: 所有模型均失败导致任务反复重新入队")
        print(f"   建议: 检查网络连接、API 配额或稍后重新运行脚本")
    else:
        print(f"🎉 所有测试已完成！结果保存在 {OUT_DIR} 目录。")


if __name__ == "__main__":
    main()
