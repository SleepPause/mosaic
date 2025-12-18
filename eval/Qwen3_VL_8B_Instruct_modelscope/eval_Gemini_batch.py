import os
import time
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
import sys

# 日志文件类
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# ================= 配置区域 =================
# 支持多个 API Key，自动轮询
GEMINI_API_KEYS = [

]

# 模型列表，支持自动降级/切换
MODELS = ["gemini-2.5-flash-lite","gemini-3-flash","gemini-2.5-flash" ]
current_model_index = 0

# 【关键回退】: 既然 Batch=20 导致错误率飙升，我们必须降回 10。
# 质量 > 数量。如果一天跑不完，宁愿跑两天，也不要一堆错误数据。
MAX_FILES_PER_BATCH = 12      
MAX_CHARS_PER_BATCH = 500000  

BASE_DIR = Path(__file__).parent
INPUT_FOLDER = BASE_DIR / "eval_text"
OUTPUT_FILE = INPUT_FOLDER / f"gemini_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
# ===========================================

# 全局变量跟踪当前 Key 索引
current_key_index = 0

def get_client():
    """获取当前使用的 OpenAI 客户端"""
    global current_key_index
    api_key = GEMINI_API_KEYS[current_key_index]
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def rotate_key():
    """切换到下一个 API Key"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    print(f"🔄 切换 API Key 到索引 [{current_key_index}] (Key: ...{GEMINI_API_KEYS[current_key_index][-6:]})")
    return get_client()

def switch_model():
    """切换到下一个模型"""
    global current_model_index
    current_model_index = (current_model_index + 1) % len(MODELS)
    new_model = MODELS[current_model_index]
    print(f"🔄 ⚠️ 所有 Key 均尝试失败，切换模型到: {new_model}")
    return new_model

# 初始化客户端
client = get_client()

def build_batch_prompt(file_batch):
    """
    构建 JSON 格式的提示词。
    """
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
    """
    解析 JSON 响应
    """
    final_output = []
    
    # 预处理：有时候模型还是会忍不住加 ```json，需要清洗掉
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    try:
        data = json.loads(cleaned_text)
        results_list = data.get("results", [])
        
        # 将结果转为字典，方便按文件名查找
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
        # JSON 解析彻底失败
        print(f"  ❌ JSON 解析失败: {e}")
        # 尝试做一个全失败的填充
        for _ in original_batch_filenames:
            final_output.append({
                "verdict": "Error",
                "score": 0,
                "reasoning": f"JSON Decode Error. Raw: {cleaned_text[:100]}...",
                "raw_json": "N/A"
            })
            
    return final_output

def main():
    global client
    log_file = BASE_DIR / f"eval_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"[*] 日志文件: {log_file}")
    print(f"[*] 已加载 {len(GEMINI_API_KEYS)} 个 API Keys")
    print(f"[*] 可用模型: {MODELS}")

    if not INPUT_FOLDER.exists():
        print(f"错误: 文件夹 '{INPUT_FOLDER}' 不存在。")
        return

    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    if not all_files:
        print("文件夹内没有找到 .txt 文件。")
        return

    print(f"找到 {len(all_files)} 个文件。")
    print(f"策略: 强制 Batch Size = {MAX_FILES_PER_BATCH} (JSON Mode)")

    # 引入 tqdm 进度条
    from tqdm import tqdm
    pbar = tqdm(total=len(all_files), unit="file", desc="总体进度")

    results = []
    processed_count = 0
    batch_index = 0

    while processed_count < len(all_files):
        batch_index += 1
        batch_data = []      
        batch_filenames = [] 
        current_batch_chars = 0 
        
        # tqdm.write 不会破坏进度条显示
        tqdm.write(f"\n🚚 正在准备第 {batch_index} 批...")

        while len(batch_data) < MAX_FILES_PER_BATCH and processed_count < len(all_files):
            fname = all_files[processed_count]
            path = INPUT_FOLDER / fname
            
            try:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='gbk') as f:
                        content = f.read()
            except Exception as e:
                tqdm.write(f"  ❌ 读取文件 {fname} 失败: {e}")
                results.append({"文件名": fname, "Verdict": "Error", "Score": 0, "Reasoning": f"Read Error: {e}", "Raw Response": "N/A"})
                processed_count += 1
                pbar.update(1) # 更新进度条
                continue

            file_len = len(content)
            
            if current_batch_chars + file_len > MAX_CHARS_PER_BATCH:
                if len(batch_data) == 0:
                    batch_data.append((fname, content))
                    batch_filenames.append(fname)
                    processed_count += 1
                    break 
                else:
                    break 
            
            batch_data.append((fname, content))
            batch_filenames.append(fname)
            current_batch_chars += file_len
            processed_count += 1
        
        if not batch_data:
            continue

        tqdm.write(f"  -> 本批装载: {len(batch_data)} 个文件 | 总字符: {current_batch_chars}")

        # 重试逻辑
        user_prompt = build_batch_prompt(batch_data)
        
        # 增加重试次数，确保能轮询所有 Key 甚至切换模型
        # 假设有 N 个 Key，M 个 Model，至少要试 N * M 次
        max_retries = len(GEMINI_API_KEYS) * len(MODELS) + 5 
        
        consecutive_key_rotations = 0 # 记录当前模型下连续切换 Key 的次数
        
        for attempt in range(max_retries):
            current_model_name = MODELS[current_model_index]
            try:
                response = client.chat.completions.create(
                    model=current_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Output strictly in JSON format."},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    temperature=0.0,
                    # 【关键】强制 JSON 模式，这比文本提示更管用
                    response_format={"type": "json_object"} 
                )
                raw_reply = response.choices[0].message.content
                
                # 解析结果
                parsed_items = parse_json_response(raw_reply, batch_filenames)
                
                for fname, item in zip(batch_filenames, parsed_items):
                    results.append({
                        "文件名": fname,
                        "Verdict": item['verdict'],
                        "Score": item['score'],
                        "Reasoning": item['reasoning'],
                        "Raw Response": item['raw_json']
                    })

                tqdm.write("  ✅ 本批处理成功！")
                consecutive_key_rotations = 0 # 成功后重置计数
                break # 成功则跳出重试循环

            except Exception as e:
                error_str = str(e)
                tqdm.write(f"  ⚠️ API 调用出错 (尝试 {attempt+1}/{max_retries}) [Model: {current_model_name}]: {error_str}")
                
                # 检查是否需要切换 Key 的错误代码
                should_switch = any(code in error_str for code in ["429", "503", "500", "403", "400", "404", "504"])
                
                if should_switch:
                    tqdm.write(f"  🚨 检测到 API 错误，正在切换 Key...")
                    client = rotate_key()
                    consecutive_key_rotations += 1
                    
                    # 如果连续切换次数超过了 Key 的总数，说明当前模型下所有 Key 都挂了
                    if consecutive_key_rotations >= len(GEMINI_API_KEYS):
                        tqdm.write(f"  ⚠️ 当前模型 {current_model_name} 所有 Key 均尝试失败。")
                        switch_model() # 切换到下一个模型
                        consecutive_key_rotations = 0 # 重置 Key 轮询计数，在新模型下重新开始
                    
                    time.sleep(1) 
                    continue 
                
                # 其他未知错误
                if attempt < max_retries - 1:
                    wait_time = 5
                    tqdm.write(f"  ⏳ 未知错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"  ❌ 最终失败: {e}")
                    for fname in batch_filenames:
                        results.append({"文件名": fname, "Verdict": "Error", "Score": 0, "Reasoning": f"API Error: {str(e)}", "Raw Response": "N/A"})
                    break

        tqdm.write(f"  💾 正在保存当前进度 ({len(results)} 条记录)...")
        try:
            df = pd.DataFrame(results)
            df = df[["文件名", "Verdict", "Score", "Reasoning", "Raw Response"]]
            df.to_excel(OUTPUT_FILE, index=False)
            tqdm.write("  ✅ 保存成功。")
        except Exception as e:
            tqdm.write(f"  ⚠️ 保存失败: {e}")
        
        # 更新进度条
        pbar.update(len(batch_data))

        if processed_count < len(all_files):
            tqdm.write("等待 10 秒冷却...")
            time.sleep(10) 

    pbar.close()
    print(f"\n全部完成！最终结果已保存到: {OUTPUT_FILE}")
    logger.close()
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
