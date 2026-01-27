#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量文本评估脚本 - CLI2API 单线程递归版本
使用 OpenAI 兼容模式通过 Gemini Manager 代理调用 Gemini API
遍历 eval_text 下的每个子目录，在每个目录内生成独立的 Excel 结果文件
"""
import os
from pathlib import Path
import time
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ========= 配置区域 =========
# Gemini Manager 代理端口 (单线程只用一个)
GEMINI_MANAGER_PORT = "4000"
GEMINI_MANAGER_PASSWORD = "123456"  # 请修改为你的密码

# 模型配置
MODEL = "gemini-3-flash-preview"

# Batch 处理配置
MAX_FILES_PER_BATCH = 12
MAX_CHARS_PER_BATCH = 500000

# 基础目录
BASE_DIR = Path(__file__).parent
EVAL_TEXT_DIR = BASE_DIR / "eval_text"

# ================================

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
    """调用 Gemini API"""
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


def process_directory(target_dir: Path):
    """处理单个目录"""
    print(f"\n{'='*60}")
    print(f"📂 进入目录: {target_dir}")
    print(f"{'='*60}")
    
    all_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
    if not all_files:
        print("   ⚠️  无 .txt 文件，跳过。")
        return

    # 确定输出文件
    # 查找最新的 gemini_batch_results*.xlsx
    existing_excel_files = list(target_dir.glob("gemini_batch_results*.xlsx"))
    output_file = None
    results_list = []
    processed_files = set()
    
    if existing_excel_files:
        output_file = max(existing_excel_files, key=lambda f: f.stat().st_mtime)
        print(f"   📄 发现已有结果文件: {output_file.name}")
        try:
            df_existing = pd.read_excel(output_file)
            if "文件名" in df_existing.columns:
                processed_files = set(df_existing["文件名"].dropna().tolist())
                # 加载已有数据以保持完整性
                for _, row in df_existing.iterrows():
                    results_list.append({
                        "文件名": row.get("文件名", ""),
                        "Verdict": row.get("Verdict", ""),
                        "Score": row.get("Score", 0),
                        "Reasoning": row.get("Reasoning", ""),
                        "Raw Response": row.get("Raw Response", "")
                    })
                print(f"   ✅ 已处理文件数: {len(processed_files)}")
            else:
                print(f"   ⚠️  警告: Excel文件中没有'文件名'列。")
        except Exception as e:
            print(f"   ❌ 读取已有Excel失败: {e}")
    else:
        output_file = target_dir / f"gemini_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"   🆕 创建新结果文件: {output_file.name}")

    # 过滤待处理文件
    files_to_process = [f for f in all_files if f not in processed_files]
    if not files_to_process:
        print("   ✨ 所有文件均已处理完毕。")
        return

    print(f"   🚀 待处理文件: {len(files_to_process)} 个")
    
    # 准备 API 客户端
    client = get_client(GEMINI_MANAGER_PORT)
    
    # 构建批次并处理
    file_index = 0
    pbar = tqdm(total=len(files_to_process), unit="file", desc="   进度")
    
    while file_index < len(files_to_process):
        # 构建当前批次
        batch_data = []
        batch_filenames = []
        current_batch_chars = 0
        
        while len(batch_data) < MAX_FILES_PER_BATCH and file_index < len(files_to_process):
            fname = files_to_process[file_index]
            path = target_dir / fname
            
            try:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='gbk') as f:
                        content = f.read()
                
                file_len = len(content)
                
                if current_batch_chars + file_len > MAX_CHARS_PER_BATCH and len(batch_data) > 0:
                    break
                
                batch_data.append((fname, content))
                batch_filenames.append(fname)
                current_batch_chars += file_len
                file_index += 1
                
            except Exception as e:
                print(f"\n   ❌ 读取文件 {fname} 失败: {e}")
                file_index += 1
        
        if not batch_data:
            continue
            
        # 处理批次
        user_prompt = build_batch_prompt(batch_data)
        
        # 简单的重试机制
        retry_count = 0
        max_retries = 3
        current_batch_success = False
        
        while retry_count < max_retries:
            response_text, success = call_api(client, user_prompt, MODEL)
            
            if success:
                parsed_items = parse_json_response(response_text, batch_filenames)
                
                # 保存结果
                batch_results_to_save = []
                for fname_res, item in zip(batch_filenames, parsed_items):
                    res_entry = {
                        "文件名": fname_res,
                        "Verdict": item['verdict'],
                        "Score": item['score'],
                        "Reasoning": item['reasoning'],
                        "Raw Response": item['raw_json']
                    }
                    batch_results_to_save.append(res_entry)
                
                results_list.extend(batch_results_to_save)
                
                # 写入 Excel
                try:
                    df = pd.DataFrame(results_list)
                    df = df[["文件名", "Verdict", "Score", "Reasoning", "Raw Response"]]
                    df.to_excel(output_file, index=False)
                except Exception as e:
                    print(f"\n   ❌ 保存 Excel 失败: {e}")
                
                pbar.update(len(batch_data))
                current_batch_success = True
                # 成功后稍作休息，避免速率限制
                time.sleep(2) 
                break
            else:
                retry_count += 1
                print(f"\n   ⚠️ API 调用失败 (重试 {retry_count}/{max_retries}): {response_text[:100]}...")
                time.sleep(5) # 失败等待
        
        if not current_batch_success:
            print(f"\n   ❌ 批次处理彻底失败，跳过 {len(batch_filenames)} 个文件")
            # 在结果中标记错误，防止死循环或丢失进度
            for fname_err in batch_filenames:
                 results_list.append({
                        "文件名": fname_err,
                        "Verdict": "Error",
                        "Score": 0,
                        "Reasoning": "Max retries exceeded",
                        "Raw Response": "N/A"
                    })
            # 尝试保存错误记录
            try:
                df = pd.DataFrame(results_list)
                df.to_excel(output_file, index=False)
            except:
                pass
            pbar.update(len(batch_filenames))

    pbar.close()
    print(f"   🏁 目录处理完成。")


def main():
    if not EVAL_TEXT_DIR.exists():
        print(f"❌ 目录 '{EVAL_TEXT_DIR}' 不存在。")
        return
    
    print(f"🚀 开始遍历目录: {EVAL_TEXT_DIR}")
    print(f"🤖 模型: {MODEL}")
    print(f"📦 Batch 大小: {MAX_FILES_PER_BATCH}")
    
    # 递归遍历所有子目录
    # os.walk 会返回 (root, dirs, files)
    # 我们只需要关注 root，看它里面有没有 .txt
    
    target_directories = []
    for root, dirs, files in os.walk(EVAL_TEXT_DIR):
        has_txt = any(f.endswith('.txt') for f in files)
        if has_txt:
            target_directories.append(Path(root))
            
    if not target_directories:
        print("未找到任何包含 .txt 文件的目录。")
        return
        
    print(f"🎯 共发现 {len(target_directories)} 个待处理目录")
    
    for i, target_dir in enumerate(target_directories, 1):
        print(f"\n[{i}/{len(target_directories)}] 正在处理目录...")
        process_directory(target_dir)
        
    print("\n✅ 所有目录处理完毕！")


if __name__ == "__main__":
    main()
