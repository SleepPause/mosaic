#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量文本评估脚本 - CLI2API 多线程版本
使用 OpenAI 兼容模式通过 Gemini Manager 代理调用 Gemini API
支持多个代理端口并发处理
"""
import os
from pathlib import Path
import time
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# ========= 配置区域 =========
# Gemini Manager 代理端口列表（每个线程使用一个端口）
GEMINI_MANAGER_PORTS = ["4000"]
GEMINI_MANAGER_PASSWORD = "123456"  # 请修改为你的密码

# 模型配置
MODEL = "gemini-3-flash-preview"

# Batch 处理配置
MAX_FILES_PER_BATCH = 12
MAX_CHARS_PER_BATCH = 500000

# 基础目录
BASE_DIR = Path(__file__).parent
INPUT_FOLDER = BASE_DIR / "eval_text"
# OUTPUT_FILE 将在 main 函数中根据已有文件动态确定
OUTPUT_FILE = None

# ================================

# 线程安全锁
save_lock = threading.Lock()
print_lock = threading.Lock()


def safe_print(msg):
    """线程安全的打印函数"""
    with print_lock:
        tqdm.write(msg)


def get_client(port):
    """获取 OpenAI 客户端（连接到 Gemini Manager）"""
    base_url = f"http://localhost:{port}/v1"
    return OpenAI(
        base_url=base_url,
        api_key=GEMINI_MANAGER_PASSWORD
    )


def build_batch_prompt(file_batch):
    """构建批量处理的提示词"""
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
    
    return prompt


def parse_json_response(response_text, original_batch_filenames):
    """解析 JSON 响应"""
    final_output = []
    cleaned_text = response_text.strip()
    
    # 移除可能的 markdown 标记
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
        print(f"  ❌ JSON 解析失败: {e}")
        for _ in original_batch_filenames:
            final_output.append({
                "verdict": "Error",
                "score": 0,
                "reasoning": f"JSON Decode Error. Raw: {cleaned_text[:100]}...",
                "raw_json": "N/A"
            })
            
    return final_output


def call_api(client, prompt, model):
    """
    调用 Gemini API
    
    Returns:
        (response_text, success): 响应文本和是否成功的标志
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Output strictly in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        
        if not answer:
            return "API Error: Empty response from API", False
        
        # 检查是否是代理错误消息
        proxy_error_indicators = [
            "所有API密钥均请求失败",
            "具体错误请查看轮询日志",
            "All API keys failed"
        ]
        
        if any(indicator in answer for indicator in proxy_error_indicators):
            return f"Proxy Error: {answer[:200]}", False
        
        return answer, True
        
    except Exception as e:
        return f"API Error: {str(e)}", False


def process_with_thread(thread_id, port, task_queue, pbar, results_list):
    """
    线程工作函数
    
    Args:
        thread_id: 线程ID
        port: Gemini Manager 端口号
        task_queue: 任务队列
        pbar: 进度条
        results_list: 结果列表（线程安全）
    """
    client = get_client(port)
    safe_print(f"🔑 Thread-{thread_id} 启动 (Port: {port})")
    
    while True:
        try:
            # 从队列获取批次任务
            try:
                batch_task = task_queue.get(timeout=0.1)
            except queue.Empty:
                break
            
            batch_data = batch_task['batch_data']
            batch_filenames = batch_task['batch_filenames']
            
            # 处理当前批次
            safe_print(f"[Thread-{thread_id}] 正在处理批次: {len(batch_data)} 个文件")
            user_prompt = build_batch_prompt(batch_data)
            
            # 调用 API
            response_text, success = call_api(client, user_prompt, MODEL)
            
            if success:
                # 解析结果
                parsed_items = parse_json_response(response_text, batch_filenames)
                
                # 构建结果条目
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
                
                # 线程安全地保存结果
                save_success = False
                with save_lock:
                    results_list.extend(batch_results)
                    
                    # 实时保存到 Excel（使用临时文件）
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
                        safe_print(f"⚠️ [Thread-{thread_id}] 保存 Excel 失败: {e}")
                        if temp_file.exists():
                            try:
                                temp_file.unlink()
                            except:
                                pass
                
                # 更新进度条
                pbar.update(len(batch_data))
                
                save_status = "✓" if save_success else "✗"
                safe_print(f"  ✅ [Thread-{thread_id}] 批次处理成功 ({len(batch_data)} 文件) [保存: {save_status}]")
                
                # 标记任务完成
                task_queue.task_done()
                
                # 成功后等待
                time.sleep(10)
            else:
                # API 调用失败，重新入队
                safe_print(f"  ❌ [Thread-{thread_id}] API 调用失败: {response_text[:200]}")
                safe_print(f"  ⏸️ [Thread-{thread_id}] 等待 60 秒后重试...")
                
                # 将批次放回队列
                task_queue.put(batch_task)
                task_queue.task_done()
                
                time.sleep(60)
                
        except Exception as e:
            safe_print(f"❌ [Thread-{thread_id}] 线程异常: {e}")
            import traceback
            safe_print(f"   Traceback: {traceback.format_exc()}")
            
            # 尝试将任务放回队列
            try:
                if 'batch_task' in locals():
                    task_queue.put(batch_task)
            except:
                pass
            
            task_queue.task_done()
            break
    
    safe_print(f"🏁 Thread-{thread_id} 任务结束")


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
    
    # 检查已处理的文件（查找所有 gemini_batch_results 开头的 Excel 文件）
    processed_files = set()
    existing_excel_files = list(INPUT_FOLDER.glob("gemini_batch_results*.xlsx"))
    
    if existing_excel_files:
        # 找到最新的Excel文件
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
        # 创建新文件
        OUTPUT_FILE = INPUT_FOLDER / f"gemini_batch_results_cli2api_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"[*] 未发现已有结果文件，将创建新文件: {OUTPUT_FILE.name}")
    
    # 过滤出未处理的文件
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("所有文件均已处理完毕，无需继续。")
        return
    
    print(f"[*] 跳过已处理文件: {len(processed_files)} 个")
    print(f"[*] 待处理文件: {len(files_to_process)} 个")
    print(f"[*] 使用模型: {MODEL}")
    print(f"[*] 并发线程数: {len(GEMINI_MANAGER_PORTS)}")
    print(f"[*] 代理端口: {', '.join(GEMINI_MANAGER_PORTS)}")
    print(f"[*] 输出文件: {OUTPUT_FILE}\n")

    # 加载已有数据
    results_list = []
    if OUTPUT_FILE.exists():
        try:
            df_existing = pd.read_excel(OUTPUT_FILE)
            for _, row in df_existing.iterrows():
                results_list.append({
                    "文件名": row.get("文件名", ""),
                    "Verdict": row.get("Verdict", ""),
                    "Score": row.get("Score", 0),
                    "Reasoning": row.get("Reasoning", ""),
                    "Raw Response": row.get("Raw Response", "")
                })
            print(f"[*] 已加载 {len(results_list)} 条已有记录\n")
        except Exception as e:
            print(f"[!] 加载已有数据失败: {e}，将从空列表开始\n")
    
    # 构建批次任务队列
    print("正在构建批次任务队列...")
    task_queue = queue.Queue()
    
    file_index = 0
    total_batches = 0
    
    while file_index < len(files_to_process):
        # 构建当前批次
        batch_data = []
        batch_filenames = []
        current_batch_chars = 0
        
        while len(batch_data) < MAX_FILES_PER_BATCH and file_index < len(files_to_process):
            fname = files_to_process[file_index]
            path = INPUT_FOLDER / fname
            
            try:
                # 读取文件内容
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='gbk') as f:
                        content = f.read()
                
                file_len = len(content)
                
                # 检查是否超过字符限制
                if current_batch_chars + file_len > MAX_CHARS_PER_BATCH and len(batch_data) > 0:
                    # 当前批次已满，不再添加
                    break
                
                batch_data.append((fname, content))
                batch_filenames.append(fname)
                current_batch_chars += file_len
                file_index += 1
                
            except Exception as e:
                print(f"\n读取文件 {fname} 失败: {e}")
                file_index += 1
        
        if batch_data:
            task_queue.put({
                'batch_data': batch_data,
                'batch_filenames': batch_filenames
            })
            total_batches += 1
    
    print(f"任务队列已构建，共 {total_batches} 个批次，{len(files_to_process)} 个文件\n")
    
    # 创建进度条
    pbar = tqdm(total=len(files_to_process), unit="file", desc="处理进度")
    
    # 启动线程池
    with ThreadPoolExecutor(max_workers=len(GEMINI_MANAGER_PORTS)) as executor:
        futures = []
        for i, port in enumerate(GEMINI_MANAGER_PORTS):
            futures.append(executor.submit(
                process_with_thread, i, port, task_queue, pbar, results_list
            ))
        
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
