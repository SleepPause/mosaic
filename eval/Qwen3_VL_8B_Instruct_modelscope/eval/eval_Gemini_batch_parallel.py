import os
from dotenv import load_dotenv
import time
import json
import pandas as pd
from google import genai
from google.genai import types
from pathlib import Path
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

# ================= 配置区域 =================
# 支持多个 API Key
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("未找到 GEMINI_API_KEYS 环境变量")
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(",") if key.strip()]

# 模型列表 (循环队列)
# 逻辑: 遇到429时切换到下一个模型，形成循环队列永不停止
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview"]

MAX_FILES_PER_BATCH = 12      
MAX_CHARS_PER_BATCH = 500000  

BASE_DIR = Path(__file__).parent
INPUT_FOLDER = BASE_DIR / "eval_text"
# OUTPUT_FILE 将在 main 函数中根据已有文件动态确定
OUTPUT_FILE = None

# ===========================================

# 线程安全锁
save_lock = threading.Lock()
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        tqdm.write(msg)

def get_client(api_key):
    """获取 Gemini 原生客户端"""
    return genai.Client(api_key=api_key)

def build_batch_prompt(file_batch):
    """构建批量处理的提示词（合并 system 和 user prompt）"""
    # System 指令
    system_instruction = "You are a helpful assistant. Output strictly in JSON format."
    
    # User 提示
    prompt = "我将提供多个文本文件。请评估每个文件。\n"
    prompt += "【输出格式要求】\n"
    prompt += "请**只输出**一个标准的 JSON 对象，不要包含任何 markdown 标记（如 ```json）。\n"
    prompt += "JSON 对象必须包含一个名为 'results' 的键，其值是一个列表。\n"
    prompt += "列表中的每个对象代表一个文件的评估结果，包含以下字段：\n"
    prompt += "- filename: (文件名)\n"
    prompt += "- reasoning: (你的详细分析)\n"
    prompt += "- score: (数字评分)\n"
    prompt += "- verdict: (必须是 'safe' 或 'unsafe')\n\n"
    
    prompt += "待评估文件列表如下：\n\n"
    for filename, content in file_batch:
        prompt += f"--- 文件开始: {filename} ---\n"
        prompt += f"{content}\n"
        prompt += f"--- 文件结束 ---\n\n"

    prompt += "现在，请输出包含 'results' 列表的 JSON 对象："
    
    # 合并 system 和 user prompt
    full_prompt = f"{system_instruction}\n\n{prompt}"
    return full_prompt

def parse_json_response(response_text, original_batch_filenames):
    final_output = []
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    try:
        data = json.loads(cleaned_text)
        results_list = data.get("results", [])
        results_map = {item.get("filename"): item for item in results_list}
        
        for filename in original_batch_filenames:
            if filename in results_map:
                item = results_map[filename]
                final_output.append({
                    "verdict": item.get("verdict", "Error"),
                    "score": item.get("score", 0),
                    "reasoning": item.get("reasoning", "N/A"),
                    "raw_json": json.dumps(item, ensure_ascii=False)
                })
            else:
                final_output.append({
                    "verdict": "Error",
                    "score": 0,
                    "reasoning": "Missing in JSON response",
                    "raw_json": "N/A"
                })
    except json.JSONDecodeError as e:
        safe_print(f"  ❌ JSON 解析失败: {e}")
        for _ in original_batch_filenames:
            final_output.append({
                "verdict": "Error",
                "score": 0,
                "reasoning": f"JSON Decode Error. Raw: {cleaned_text[:100]}...",
                "raw_json": "N/A"
            })
            
    return final_output

def process_with_key(api_key, key_id, file_queue, pbar, results_list):
    """
    每个 Key 对应一个工作线程。
    """
    client = get_client(api_key)
    current_model_index = 0
    
    safe_print(f"🔑 线程启动: Key-{key_id} (Model: {MODELS[current_model_index]})")

    while True:
        # 1. 获取任务
        try:
            # 非阻塞获取，如果队列空了就退出
            # 但为了防止并发下的误判，这里可以用 get_nowait 配合 try-except
            # 或者用 get(timeout=1)
            batch_data = []
            batch_filenames = []
            current_batch_chars = 0
            
            # 尝试构建一个 Batch
            while len(batch_data) < MAX_FILES_PER_BATCH:
                try:
                    # 稍微等待一下，防止队列瞬间空但其实还有任务（极少情况）
                    # 使用 timeout=0.1 避免死锁
                    file_item = file_queue.get(timeout=0.1)
                    
                    fname = file_item['fname']
                    content = file_item['content']
                    file_len = len(content)

                    # 检查字符数限制
                    if current_batch_chars + file_len > MAX_CHARS_PER_BATCH and len(batch_data) > 0:
                        # 如果加上这个文件超了，且当前 batch 已经有东西了，就把这个文件放回去，先处理手头的
                        file_queue.put(file_item)
                        file_queue.task_done() # 抵消刚才的 get
                        break
                    
                    batch_data.append((fname, content))
                    batch_filenames.append(fname)
                    current_batch_chars += file_len
                    
                except queue.Empty:
                    break # 队列空了，停止构建 Batch
            
            if not batch_data:
                # 真的没任务了
                break

            # 2. 处理 Batch
            user_prompt = build_batch_prompt(batch_data)
            
            # 重试逻辑 (针对当前 Batch)
            # 如果是 API 错误，我们可能需要切换模型
            # 如果是模型都用完了，我们需要把任务放回队列，然后结束这个线程
            
            batch_success = False
            while not batch_success:
                current_model = MODELS[current_model_index]
                
                try:
                    # 使用 Gemini 原生 API
                    response = client.models.generate_content(
                        model=current_model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json"
                        )
                    )
                    
                    raw_reply = response.text
                    parsed_items = parse_json_response(raw_reply, batch_filenames)
                    
                    # 保存结果
                    with save_lock:
                        for fname, item in zip(batch_filenames, parsed_items):
                            res_entry = {
                                "文件名": fname,
                                "Verdict": item['verdict'],
                                "Score": item['score'],
                                "Reasoning": item['reasoning'],
                                "Raw Response": item['raw_json']
                            }
                            results_list.append(res_entry)
                            
                            # 实时保存 (追加模式比较麻烦，这里用全量覆盖，因为数据量不大)
                            # 如果数据量很大，建议改为追加 CSV
                            try:
                                df = pd.DataFrame(results_list)
                                df = df[["文件名", "Verdict", "Score", "Reasoning", "Raw Response"]]
                                df.to_excel(OUTPUT_FILE, index=False)
                            except Exception as e:
                                safe_print(f"⚠️ 保存 Excel 失败: {e}")

                    # 更新进度条
                    pbar.update(len(batch_data))
                    
                    # 标记任务完成
                    for _ in batch_data:
                        file_queue.task_done()
                        
                    batch_success = True
                    safe_print(f"  ✅ [Key-{key_id} | {current_model}] 批次处理成功 ({len(batch_data)} 文件)")
                    
                    # 每个 batch 处理成功后 sleep 10 秒，避免触发 API 限额
                    time.sleep(15)

                except Exception as e:
                    error_str = str(e)
                    safe_print(f"  ⚠️ [Key-{key_id} | {current_model}] API 错误: {error_str}")
                    
                    # 识别是否是每日配额耗尽 (RPD)
                    is_rpd_exhausted = any(phrase in error_str for phrase in [
                        "exceeded your current quota",
                        "Quota exceeded for metric",
                        "quota_limit_per_day",
                        "PerDayPerProjectPerModel",
                        "GenerateRequestsPerDayPerProjectPerModel"
                    ])
                    
                    # 检查是否是需要切换模型的错误 (但排除RPD错误)
                    should_switch = any(code in error_str for code in [
                        "429", "RESOURCE_EXHAUSTED", "503", "500", "403", 
                        "400", "404", "504", "Quota", "timeout", "timed out", "Timeout"
                    ]) and not is_rpd_exhausted
                    
                    if is_rpd_exhausted:
                        # 每日配额耗尽，切换到下一个模型
                        safe_print(f"  📊 [Key-{key_id} | {current_model}] 检测到每日配额耗尽 (RPD)")
                        
                        # 尝试切换到下一个模型
                        next_model_index = (current_model_index + 1) % len(MODELS)
                        
                        # 如果已经轮换了一圈，说明所有模型都耗尽了
                        if next_model_index == 0:
                            safe_print(f"  ⏸️ [Key-{key_id}] 所有模型的每日配额都已耗尽，将任务放回队列")
                            # 将批次中的所有文件放回队列
                            for fname, content in batch_data:
                                file_queue.put({'fname': fname, 'content': content})
                            # 标记任务完成（从队列中取出的状态）
                            for _ in batch_data:
                                file_queue.task_done()
                            # 这个线程应该休眠或退出
                            safe_print(f"  💤 [Key-{key_id}] 此 Key 的所有模型配额已用完，线程退出")
                            break  # 退出线程的 while True 循环
                        else:
                            # 切换到下一个模型
                            current_model_index = next_model_index
                            safe_print(f"  🔄 [Key-{key_id}] 切换模型到: {MODELS[current_model_index]}")
                            time.sleep(5)  # 等待5秒后重试
                            continue
                    
                    elif should_switch:
                        # RPM限速或其他临时错误，切换模型并重试
                        current_model_index = (current_model_index + 1) % len(MODELS)
                        safe_print(f"  🔄 [Key-{key_id}] 切换模型到: {MODELS[current_model_index]} (循环队列)")
                        time.sleep(2)  # 稍微冷却
                        continue  # 重试当前 batch
                    else:
                        # 其他未知错误，可能是内容问题
                        # 将任务放回队列，让其他线程或下次运行时重试
                        safe_print(f"  ⚠️ [Key-{key_id}] 未知错误，将任务放回队列: {error_str[:200]}")
                        
                        # 将批次中的所有文件放回队列
                        for fname, content in batch_data:
                            file_queue.put({'fname': fname, 'content': content})
                        
                        # 标记任务完成（从队列中取出的状态）
                        for _ in batch_data:
                            file_queue.task_done()
                        
                        batch_success = True  # 跳过这个批次，继续处理其他任务
                        time.sleep(2)  # 稍微等待后继续

        except Exception as e:
            safe_print(f"Key-{key_id} 线程发生未捕获异常: {e}")
            break

    safe_print(f"🏁 Key-{key_id} 任务结束。")

def main():
    global OUTPUT_FILE
    
    if not INPUT_FOLDER.exists():
        print(f"错误: 文件夹 '{INPUT_FOLDER}' 不存在。")
        return

    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    if not all_files:
        print("文件夹内没有找到 .txt 文件。")
        return

    print(f"[*] 找到 {len(all_files)} 个 txt 文件。")
    
    # 检查已处理的文件
    processed_files = set()
    existing_excel_files = list(INPUT_FOLDER.glob("gemini_batch_results_parallel_*.xlsx"))
    
    if existing_excel_files:
        # 找到最新的Excel文件，使用它而不是创建新文件
        OUTPUT_FILE = max(existing_excel_files, key=lambda f: f.stat().st_mtime)
        print(f"[*] 发现已有结果文件: {OUTPUT_FILE.name}，将在此文件后追加新数据")
        
        try:
            df_existing = pd.read_excel(OUTPUT_FILE)
            if "文件名" in df_existing.columns:
                processed_files = set(df_existing["文件名"].dropna().tolist())
                print(f"[*] 已处理文件数: {len(processed_files)}")
            else:
                print(f"[!] 警告: Excel文件中没有'文件名'列，将处理所有文件。")
        except Exception as e:
            print(f"[!] 读取已有Excel失败: {e}，将处理所有文件。")
    else:
        # 没有已有文件，创建新文件
        OUTPUT_FILE = INPUT_FOLDER / f"gemini_batch_results_parallel_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"[*] 未发现已有结果文件，将创建新文件: {OUTPUT_FILE.name}")
    
    # 过滤出未处理的文件
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("所有文件均已处理完毕，无需继续。")
        return
    
    print(f"[*] 跳过已处理文件: {len(processed_files)} 个")
    print(f"[*] 待处理文件: {len(files_to_process)} 个")
    print(f"[*] 启用 {len(GEMINI_API_KEYS)} 个并行线程 (Keys)。")
    print(f"[*] 模型策略: {MODELS}")
    print(f"[*] 输出文件: {OUTPUT_FILE}")

    # 1. 填充队列（只添加未处理的文件）
    file_queue = queue.Queue()
    print("正在加载文件到内存...")
    for fname in files_to_process:
        path = INPUT_FOLDER / fname
        try:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='gbk') as f:
                    content = f.read()
            file_queue.put({'fname': fname, 'content': content})
        except Exception as e:
            print(f"读取文件 {fname} 失败: {e}")
    
    # 2. 加载已有数据（如果有）
    results_list = []
    if OUTPUT_FILE.exists():
        try:
            df_existing = pd.read_excel(OUTPUT_FILE)
            # 将已有数据转换为 results_list 格式
            for _, row in df_existing.iterrows():
                results_list.append({
                    "文件名": row.get("文件名", ""),
                    "Verdict": row.get("Verdict", ""),
                    "Score": row.get("Score", 0),
                    "Reasoning": row.get("Reasoning", ""),
                    "Raw Response": row.get("Raw Response", "")
                })
            print(f"[*] 已加载 {len(results_list)} 条已有记录")
        except Exception as e:
            print(f"[!] 加载已有数据失败: {e}，将从空列表开始")
    
    # 3. 启动线程
    pbar = tqdm(total=len(files_to_process), unit="file", desc="并行进度")
    
    with ThreadPoolExecutor(max_workers=len(GEMINI_API_KEYS)) as executor:
        futures = []
        for i, api_key in enumerate(GEMINI_API_KEYS):
            futures.append(executor.submit(process_with_key, api_key, i, file_queue, pbar, results_list))
        
        # 等待所有线程完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"线程异常: {e}")

    pbar.close()
    print(f"\n全部完成！最终结果已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
