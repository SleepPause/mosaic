#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
使用 ModelScope API 实现（Key轮询版本，支持 thinking）
文件名：图片名_策略名_序号_qwen_时间戳.txt

【Key轮询版本】n个工作线程从l个API Key的循环队列中轮询获取key，每个任务依次尝试模型列表
"""
import time
import re
import logging
import random
import signal
import collections
from datetime import datetime
from pathlib import Path
import prompt_strategies_new as prompt_strategies  # 导入新版提示词库
import evaluation_templates
import os
from dotenv import load_dotenv
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import base64
import httpx

load_dotenv()

# ========= 用户配置区域 =========
# 获取 ModelScope API Keys
DASHSCOPE_API_KEYS_STR = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEYS_STR:
    raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量")
DASHSCOPE_API_KEYS = [key.strip() for key in DASHSCOPE_API_KEYS_STR.split(",") if key.strip()]

# ModelScope API 配置
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODELS = ["Qwen/Qwen3-VL-30B-A3B-Thinking"]  # 模型列表（支持扩展）

# 工作线程数配置
NUM_WORKERS = 5  # 固定5个工作线程

# 使用脚本所在目录作为基准，防止在不同目录下运行时找不到文件
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"
OUT_DIR = BASE_DIR / "eval_text1"  # 修改输出目录以区分

# 每个策略重复测试的次数
REPEAT_TIMES = 10

# 速率限制配置
SUCCESS_SLEEP_MIN = 10  # 成功后最少等待10秒
SUCCESS_SLEEP_MAX = 15  # 成功后最多等待15秒（加入随机性）
FAILURE_SLEEP = 20       # 单个模型失败后等待5秒再尝试下一个模型

# 配额监控阈值
QUOTA_WARNING_THRESHOLD = 0.10  # 低于10%发出警告
QUOTA_CRITICAL_THRESHOLD = 0.05  # 低于5%严重警告
# =================================

# 使用 StrongREJECT 模板
EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT

# 线程安全锁
print_lock = threading.Lock()
stats_lock = threading.Lock()
quota_lock = threading.Lock()

# 统计计数器
stats = {
    'completed': 0,    # 成功完成的任务
    'skipped': 0,      # 跳过的任务（已存在）
    'failed': 0        # 失败的任务（重新入队）
}

# 配额状态追踪（key_id -> quota_info）
key_quota_status = {}

# 优雅退出标志
stop_flag = False

# 日志系统
logger = None  # 将在 main 函数中初始化


class KeyQueue:
    """线程安全的API Key循环队列，支持配额耗尽的 key 自动移除"""
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
        # 配额耗尽的 key_id 黑名单
        self.exhausted_keys = set()
    
    def get_key(self) -> tuple:
        """从队列头部获取一个未耗尽的key
        
        Returns:
            (key_id, api_key) 或 (None, None) 如果队列为空
        """
        with self.lock:
            if not self.queue:
                return None, None
            key_tuple = self.queue.popleft()
            return key_tuple
    
    def return_key(self, key_id: int, api_key: str):
        """将key放回队列末尾（仅当未被标记为耗尽时）
        
        Args:
            key_id: Key的ID
            api_key: API Key字符串
        """
        with self.lock:
            if key_id not in self.exhausted_keys:
                self.queue.append((key_id, api_key))
    
    def mark_key_exhausted(self, key_id: int):
        """标记某个 key 的配额已耗尽
        
        Args:
            key_id: 要标记的 Key ID
        """
        with self.lock:
            self.exhausted_keys.add(key_id)
            log_and_print(
                f"🚫 Key-{key_id} 配额已耗尽，已从轮询队列中移除",
                level='warning',
                console=True
            )
    
    def size(self) -> int:
        """获取当前队列大小"""
        with self.lock:
            return len(self.queue)
    
    def get_exhausted_count(self) -> int:
        """获取已耗尽的 key 数量"""
        with self.lock:
            return len(self.exhausted_keys)


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
    log_file = log_dir / f"eval_qwen_polling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logger = logging.getLogger('QwenPollingEval')
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


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    suffix = image_path.suffix.lower().strip(".")
    if suffix == 'jpg':
        suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def safe_file_name(name: str) -> str:
    """去掉特殊符号"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def get_quota_icon(remaining_percent: float) -> str:
    """根据剩余配额百分比返回状态图标"""
    if remaining_percent > 0.5:
        return "🟢"
    elif remaining_percent > 0.25:
        return "🟡"
    else:
        return "🔴"


def update_key_quota(key_id: int, quota_info: dict):
    """更新并检查配额状态"""
    with quota_lock:
        key_quota_status[key_id] = quota_info
        
        try:
            model_remaining = int(quota_info.get("模型剩余额度", 0))
            model_limit = int(quota_info.get("模型当天限额", 1))
            remaining_percent = model_remaining / model_limit if model_limit > 0 else 0
            
            quota_info['remaining_percent'] = remaining_percent
            
            # 发出警告
            if remaining_percent < QUOTA_CRITICAL_THRESHOLD:
                log_and_print(
                    f"🚨 Key-{key_id} 配额严重不足: {model_remaining}/{model_limit} ({remaining_percent*100:.1f}%)",
                    level='warning',
                    console=True
                )
            elif remaining_percent < QUOTA_WARNING_THRESHOLD:
                log_and_print(
                    f"⚠️  Key-{key_id} 配额预警: {model_remaining}/{model_limit} ({remaining_percent*100:.1f}%)",
                    level='warning'
                )
        except (ValueError, ZeroDivisionError):
            pass


def get_response(api_key: str, text_prompt: str, image_path: Path, model: str, key_id: int) -> tuple:
    """
    调用 ModelScope API 获取回答（流式响应，支持 thinking）
    使用 httpx 直接调用以获取响应头进行配额监控
    
    Returns:
        (answer, is_quota_exhausted, reasoning, quota_info)
        is_quota_exhausted: True 表示该 key 配额已完全耗尽
    """
    image_b64 = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
        "stream": True,
        "extra_body": {
            'enable_thinking': True,
            "thinking_budget": 81920
        }
    }
    
    answer_content = ""
    reasoning_content = ""
    quota_info = None
    
    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{BASE_URL}/chat/completions", headers=headers, json=payload) as response:
                
                # 提取响应头（配额信息）
                quota_info = {
                    "用户当天限额": response.headers.get("modelscope-ratelimit-requests-limit", "N/A"),
                    "用户剩余额度": response.headers.get("modelscope-ratelimit-requests-remaining", "N/A"),
                    "模型当天限额": response.headers.get("modelscope-ratelimit-model-requests-limit", "N/A"),
                    "模型剩余额度": response.headers.get("modelscope-ratelimit-model-requests-remaining", "N/A"),
                }
                
                # 检查配额是否耗尽（响应头方式）
                try:
                    model_remaining = int(quota_info.get("模型剩余额度", -1))
                    if model_remaining == 0:
                        log_and_print(
                            f"[Key-{key_id}] 检测到模型配额耗尽（响应头显示剩余0）",
                            level='warning',
                            console=True
                        )
                        return "API Error: Quota exhausted (remaining=0)", True, "", quota_info
                except (ValueError, TypeError):
                    pass
                
                # 更新配额状态
                update_key_quota(key_id, quota_info)
                
                if response.status_code != 200:
                    error_text = response.text
                    log_and_print(f"[Key-{key_id}] HTTP错误 {response.status_code}: {error_text}", level='warning')
                    return f"HTTP Error {response.status_code}: {error_text}", False, "", quota_info
                
                # 处理流式响应
                import json
                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        line = line[6:]  # 移除 "data: " 前缀
                    
                    try:
                        chunk_data = json.loads(line)
                        if "choices" not in chunk_data or not chunk_data["choices"]:
                            continue
                        
                        delta = chunk_data["choices"][0].get("delta", {})
                        
                        # 处理 reasoning_content
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            if len(reasoning_content) < 20000:
                                reasoning_content += delta["reasoning_content"]
                            elif not reasoning_content.endswith("... [Truncated]"):
                                reasoning_content += "\n... [Thinking Process Truncated]"
                        
                        # 处理 content
                        if "content" in delta and delta["content"]:
                            answer_content += delta["content"]
                    
                    except json.JSONDecodeError:
                        continue
        
        answer = answer_content.strip()
        reasoning = reasoning_content.strip()
        
        if not answer:
            return "API Error: Empty response from API", False, "", quota_info
        
        return answer, False, reasoning, quota_info  # 成功
        
    except Exception as e:
        error_str = str(e)
        log_and_print(f"[Key-{key_id}] API错误 ({model}): {error_str}", level='warning')
        
        # 识别配额耗尽（错误信息方式）
        is_quota_exhausted = any(phrase in error_str.lower() for phrase in [
            "exceeded your current quota",
            "quota exceeded",
            "quota_limit_per_day",
            "resource has been exhausted",
            "rate limit",
            "请求过于频繁",
            "配额不足"
        ])
        
        if is_quota_exhausted:
            log_and_print(
                f"[Key-{key_id}] 检测到配额耗尽（错误信息：{error_str[:100]}）",
                level='warning',
                console=True
            )
        
        return f"API Error: {error_str}", is_quota_exhausted, "", quota_info


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
        
        # 2. 对每个模型，独立获取并处理一个任务
        for model_index, model in enumerate(MODELS):
            if stop_flag:
                break
            
            # 2.1 循环获取任务，跳过已完成的，直到找到未完成的任务
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
                
                # 修改模型标识为简短版本
                model_short = "qwen3-vl-30b-a3b"
                file_prefix = f"{safe_file_name(img_stem)}_{strat_name}_{repeat_index}_{model_short}{method_suffix}"
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
            
            # 2.2 用当前模型处理这个未完成的任务
            log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] 开始: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次")
            start_t = time.time()
            
            answer, is_quota_exhausted, reasoning, quota_info = get_response(
                api_key, prompt_text, img_path, model, key_id
            )
            duration = time.time() - start_t
            
            # 2.3 检查配额是否耗尽
            if is_quota_exhausted:
                # 标记该 key 为已耗尽，不再使用
                key_queue.mark_key_exhausted(key_id)
                log_and_print(
                    f"[Thread-{thread_id}|Key-{key_id}] 配额已耗尽，任务重新入队，Key不再归还",
                    level='warning',
                    console=True
                )
                task_queue.put(task)
                with stats_lock:
                    stats['failed'] += 1
                if task_acquired:
                    task_queue.task_done()
                # 不归还 key，直接跳出模型循环
                break
            
            # 2.4 判断结果：成功 or 失败
            if not answer.startswith("API Error:") and not answer.startswith("HTTP Error"):
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
                
                # 生成配额信息字符串
                quota_str = ""
                if quota_info:
                    try:
                        model_remaining = int(quota_info.get("模型剩余额度", 0))
                        model_limit = int(quota_info.get("模型当天限额", 1))
                        remaining_percent = model_remaining / model_limit if model_limit > 0 else 0
                        icon = get_quota_icon(remaining_percent)
                        quota_str = f" | 配额: {icon} {model_remaining}/{model_limit} ({remaining_percent*100:.1f}%)"
                    except (ValueError, ZeroDivisionError):
                        pass
                
                log_and_print(f"[Thread-{thread_id}|Key-{key_id}|{model}] ✅ 完成: {img_name} ({target_word}) | {strat_name} | 第{repeat_index}次 ({duration:.2f}s){quota_str}")
                
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
            
            # 2.5 标记任务完成
            if task_acquired:
                task_queue.task_done()

        
        # 3. 模型都处理完后，将key放回队列末尾（如果未被标记为耗尽）
        if key_id not in key_queue.exhausted_keys:
            key_queue.return_key(key_id, api_key)
            log_and_print(f"[Thread-{thread_id}] Key-{key_id} 已归还到队列（处理了{len(MODELS)}个模型任务）")
        else:
            log_and_print(f"[Thread-{thread_id}] Key-{key_id} 已耗尽，不再归还")
    
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
    print(f"[*] API Keys 数量: {len(DASHSCOPE_API_KEYS)}")
    print(f"[*] 工作线程数: {NUM_WORKERS}")
    print(f"[*] 🔄 Key轮询模式: 启用")
    print(f"[*] 📊 配额监控: 启用")
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
    key_queue = KeyQueue(DASHSCOPE_API_KEYS)
    
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
    print(f"🚫 配额耗尽的Keys: {key_queue.get_exhausted_count()} 个")
    
    # 打印配额摘要
    if key_quota_status:
        print(f"\n{'='*60}")
        print(f"📊 配额状态摘要:")
        print(f"{'='*60}")
        for key_id, quota in key_quota_status.items():
            try:
                remaining = int(quota.get("模型剩余额度", 0))
                limit = int(quota.get("模型当天限额", 1))
                percent = remaining / limit if limit > 0 else 0
                icon = get_quota_icon(percent)
                print(f"Key-{key_id}: {icon} {remaining}/{limit} ({percent*100:.1f}%)")
            except (ValueError, ZeroDivisionError):
                print(f"Key-{key_id}: 配额信息不可用")
    
    print(f"{'='*60}\n")
    
    if remaining_tasks > 0:
        print(f"⚠️ 警告: 所有线程已结束，但队列中还有 {remaining_tasks} 个未完成的任务")
        print(f"   可能原因: 所有模型均失败导致任务反复重新入队")
        print(f"   建议: 检查网络连接、API 配额或稍后重新运行脚本")
    else:
        print(f"🎉 所有测试已完成！结果保存在 {OUT_DIR} 目录。")


if __name__ == "__main__":
    main()
