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

# 模型列表
# 逻辑: 遇到配额错误(RPD/RPM)时切换到下一个模型，所有模型都耗尽后线程终止
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
    exhausted_models = set()  # 追踪已耗尽配额的模型
    should_exit_thread = False  # 线程退出标志
    
    safe_print(f"🔑 线程启动: Key-{key_id} (Model: {MODELS[current_model_index]})")

    while True:
        # 1. 获取任务
        try:
            batch_data = []
            batch_filenames = []
            current_batch_chars = 0
            
            # 尝试构建一个 Batch
            while len(batch_data) < MAX_FILES_PER_BATCH:
                try:
                    # 使用 timeout=0.1 避免死锁
                    file_item = file_queue.get(timeout=0.1)
                    
                    fname = file_item['fname']
                    content = file_item['content']
                    file_len = len(content)

                    # 检查字符数限制
                    if current_batch_chars + file_len > MAX_CHARS_PER_BATCH and len(batch_data) > 0:
                        # 如果加上这个文件超了，且当前 batch 已经有东西了
                        # 不放回队列，而是直接处理当前批次，下次循环再取这个任务
                        file_queue.put(file_item)
                        # 注意：这里不调用 task_done()，因为我们立即又放回去了
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
            
            batch_success = False
            failed_models = set()  # 追踪当前batch中失败的模型
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
                    
                    # 构建结果条目（在锁外准备）
                    batch_results = []
                    for fname, item in zip(batch_filenames, parsed_items):
                        res_entry = {
                            "文件名": fname,
                            "Verdict": item['verdict'],
                            "Score": item['score'],
                            "Reasoning": item['reasoning'],
                            "Raw Response": item['raw_json']
                        }
                        batch_results.append(res_entry)
                    
                    # 保存结果（带锁保护）
                    save_success = False
                    with save_lock:
                        results_list.extend(batch_results)
                        
                        # 实时保存到 Excel（优化：使用临时文件）
                        temp_file = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_temp.xlsx"
                        try:
                            df = pd.DataFrame(results_list)
                            df = df[["文件名", "Verdict", "Score", "Reasoning", "Raw Response"]]
                            # 先写入临时文件
                            df.to_excel(temp_file, index=False)
                            # 成功后替换原文件
                            if temp_file.exists():
                                temp_file.replace(OUTPUT_FILE)
                            save_success = True
                        except Exception as e:
                            safe_print(f"⚠️ [Key-{key_id}] 保存 Excel 失败: {e}")
                            # 如果临时文件存在，尝试删除
                            if temp_file.exists():
                                try:
                                    temp_file.unlink()
                                except:
                                    pass
                    
                    # 更新进度条（移到锁外，避免阻塞）
                    pbar.update(len(batch_data))
                    
                    # 标记任务完成
                    for _ in batch_data:
                        file_queue.task_done()
                        
                    batch_success = True
                    save_status = "✓" if save_success else "✗"
                    safe_print(f"  ✅ [Key-{key_id} | {current_model}] 批次处理成功 ({len(batch_data)} 文件) [保存: {save_status}]")
                    
                    # 每个 batch 处理成功后 sleep，避免触发 API 限额
                    time.sleep(15)

                except Exception as e:
                    error_str = str(e)
                    
                    # 识别是否是每日配额耗尽 (RPD)
                    is_rpd_exhausted = any(phrase in error_str for phrase in [
                        "exceeded your current quota",
                        "Quota exceeded for metric",
                        "quota_limit_per_day",
                        "PerDayPerProjectPerModel",
                        "GenerateRequestsPerDayPerProjectPerModel"
                    ])
                    
                    # 区分临时错误和永久错误
                    # 临时错误：可以通过切换模型或重试解决
                    is_temporary_error = any(code in error_str for code in [
                        "429", "RESOURCE_EXHAUSTED", "503", "504", 
                        "timeout", "timed out", "Timeout", "DeadlineExceeded"
                    ])
                    
                    # 永久错误：切换模型也无法解决
                    is_permanent_error = any(code in error_str for code in [
                        "400", "INVALID_ARGUMENT", "404", "NOT_FOUND"
                    ]) and not is_rpd_exhausted
                    
                    if is_rpd_exhausted:
                        # 每日配额耗尽，标记当前模型为已耗尽
                        exhausted_models.add(current_model)
                        safe_print(f"  📊 [Key-{key_id} | {current_model}] 检测到每日配额耗尽 (RPD) [已耗尽: {len(exhausted_models)}/{len(MODELS)}]")
                        
                        # 检查是否所有模型都已耗尽
                        if len(exhausted_models) >= len(MODELS):
                            safe_print(f"  ⏸️ [Key-{key_id}] 所有模型的每日配额都已耗尽")
                            # 将任务放回队列，让其他 Key 的线程处理
                            for fname, content in batch_data:
                                file_queue.put({'fname': fname, 'content': content})
                            # 线程退出（不调用task_done，因为任务已放回队列）
                            safe_print(f"  💤 [Key-{key_id}] 此 Key 的所有模型配额已用完，任务已放回队列，线程退出")
                            should_exit_thread = True
                            break  # 退出 while not batch_success 循环
                        else:
                            # 切换到下一个未耗尽的模型
                            for _ in range(len(MODELS)):
                                current_model_index = (current_model_index + 1) % len(MODELS)
                                next_model = MODELS[current_model_index]
                                if next_model not in exhausted_models:
                                    safe_print(f"  🔄 [Key-{key_id}] 切换模型到: {next_model}")
                                    break
                            time.sleep(5)  # 等待后重试
                            continue
                    
                    else:
                        # 其他错误（临时、永久、未知）：标记当前模型失败，尝试下一个
                        failed_models.add(current_model)
                        
                        # 判断错误类型
                        if is_temporary_error:
                            error_type = "临时错误"
                        elif is_permanent_error:
                            error_type = "永久错误"
                        else:
                            error_type = "未知错误"
                        
                        safe_print(f"  ⚠️ [Key-{key_id} | {current_model}] {error_type}: {error_str[:150]}")
                        safe_print(f"  📊 [Key-{key_id}] 当前batch已失败模型: {len(failed_models)}/{len(MODELS)}")
                        
                        # 检查是否所有模型都失败了
                        if len(failed_models) >= len(MODELS):
                            safe_print(f"  ❌ [Key-{key_id}] 所有模型对此batch都失败，线程退出")
                            # 将任务放回队列，让其他线程尝试
                            for fname, content in batch_data:
                                file_queue.put({'fname': fname, 'content': content})
                            safe_print(f"  💤 [Key-{key_id}] 任务已放回队列，线程退出")
                            should_exit_thread = True
                            break  # 退出 while not batch_success 循环
                        else:
                            # 切换到下一个未失败的模型
                            for _ in range(len(MODELS)):
                                current_model_index = (current_model_index + 1) % len(MODELS)
                                next_model = MODELS[current_model_index]
                                if next_model not in failed_models and next_model not in exhausted_models:
                                    safe_print(f"  🔄 [Key-{key_id}] 切换模型到: {next_model}")
                                    break
                            time.sleep(2)
                            continue

        except Exception as e:
            safe_print(f"❌ Key-{key_id} 线程发生未捕获异常: {e}")
            import traceback
            safe_print(f"   Traceback: {traceback.format_exc()}")
            break
        
        # 检查是否需要退出线程
        if should_exit_thread:
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
