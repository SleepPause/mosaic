#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有API Key的可用性（多线程并发版本）
对每个key发送"hello"消息,记录完整回答,最后生成Excel表格
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# 获取所有API Keys
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("未找到 GEMINI_API_KEYS 环境变量")

GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(",") if key.strip()]

# 测试使用的模型
MODEL = "gemma-3-12b"

# 测试消息
TEST_MESSAGE = "hello"

# 线程安全锁
results_lock = threading.Lock()
print_lock = threading.Lock()

# 存储结果
results = []

def test_api_key(api_key, key_id, pbar):
    """测试单个API key"""
    try:
        # 创建客户端
        client = genai.Client(api_key=api_key)
        
        # 发送请求
        response = client.models.generate_content(
            model=MODEL,
            contents=TEST_MESSAGE,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        
        # 提取回答
        answer = response.text.strip() if response.text else "无回答内容"
        
        with print_lock:
            tqdm.write(f"✅ [Key-{key_id}] 成功 - 回答长度: {len(answer)} 字符")
        
        # 记录结果
        with results_lock:
            results.append({
                "序号": key_id,
                "Key值": api_key[:20] + "..." + api_key[-10:],  # 只显示部分key保护隐私
                "完整回答": answer
            })
        
    except Exception as e:
        error_msg = str(e)
        
        with print_lock:
            tqdm.write(f"❌ [Key-{key_id}] 失败 - 错误: {error_msg[:200]}")
        
        # 记录错误
        with results_lock:
            results.append({
                "序号": key_id,
                "Key值": api_key[:20] + "..." + api_key[-10:],
                "完整回答": f"[错误] {error_msg}"
            })
    
    finally:
        pbar.update(1)


def main():
    """主函数"""
    print(f"找到 {len(GEMINI_API_KEYS)} 个API Keys")
    print(f"测试模型: {MODEL}")
    print(f"测试消息: {TEST_MESSAGE}")
    print(f"并发线程数: {len(GEMINI_API_KEYS)}\n")
    print("="*60)
    
    # 创建进度条
    pbar = tqdm(total=len(GEMINI_API_KEYS), unit="key", desc="测试进度", ncols=80)
    
    # 使用线程池并发测试
    with ThreadPoolExecutor(max_workers=len(GEMINI_API_KEYS)) as executor:
        futures = []
        for idx, api_key in enumerate(GEMINI_API_KEYS, start=1):
            future = executor.submit(test_api_key, api_key, idx, pbar)
            futures.append(future)
        
        # 等待所有线程完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"线程异常: {e}")
    
    pbar.close()
    
    print("\n" + "="*60)
    print("测试完成，正在生成Excel...")
    
    # 按序号排序结果
    results.sort(key=lambda x: x["序号"])
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"api_keys_test_result_{timestamp}.xlsx"
    
    # 保存为Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"✅ Excel文件已生成: {output_file}")
    print(f"\n总计测试: {len(GEMINI_API_KEYS)} 个keys")
    print(f"成功: {sum(1 for r in results if not r['完整回答'].startswith('[错误]'))} 个")
    print(f"失败: {sum(1 for r in results if r['完整回答'].startswith('[错误]'))} 个")


if __name__ == "__main__":
    main()
