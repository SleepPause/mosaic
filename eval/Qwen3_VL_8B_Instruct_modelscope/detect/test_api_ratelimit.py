#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程测试千问API的响应头，监控剩余额度
通过HTTP响应头来查看API限流信息
每个 API key 对应一个线程进行并发测试
"""
import base64
from pathlib import Path
import os
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 加载 .env 文件
load_dotenv()

# ========= API 配置 =========
API_KEYS_STR = os.getenv("DASHSCOPE_API_KEY")
if not API_KEYS_STR:
    raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件")

# 解析多个API keys（逗号分隔）
API_KEYS = [k.strip() for k in API_KEYS_STR.split(',') if k.strip()]
if not API_KEYS:
    raise ValueError("DASHSCOPE_API_KEY 环境变量为空")

BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking"

# 使用脚本所在目录作为基准
BASE_DIR = Path(__file__).parent

# 测试问题
QUESTION = "图中最大的单词是什么？只需要回答单词，不要多于输出"

# 线程锁，用于保护控制台输出
print_lock = threading.Lock()
# ============================


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    suffix = image_path.suffix.lower().strip(".")
    if suffix == 'jpg':
        suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def test_single_key(api_key: str, key_index: int, total_keys: int, text_prompt: str, image_path: Path):
    """
    测试单个 API key 的限流信息
    
    Args:
        api_key: API密钥
        key_index: 密钥索引（从1开始）
        total_keys: 总密钥数量
        text_prompt: 提示词
        image_path: 测试图片路径
    
    Returns:
        dict: 包含测试结果的字典
    """
    import httpx
    
    key_prefix = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
    
    with print_lock:
        print(f"[{key_index}/{total_keys}] 正在测试 Key: {key_prefix}")
    
    image_b64 = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }]
    }
    
    result = {
        "key_index": key_index,
        "key_prefix": key_prefix,
        "success": False,
        "response_text": None,
        "rate_limit_info": None,
        "error": None
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = httpx.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # 提取响应头
            rate_limit_info = {
                "用户当天限额": response.headers.get("modelscope-ratelimit-requests-limit", "N/A"),
                "用户剩余额度": response.headers.get("modelscope-ratelimit-requests-remaining", "N/A"),
                "模型当天限额": response.headers.get("modelscope-ratelimit-model-requests-limit", "N/A"),
                "模型剩余额度": response.headers.get("modelscope-ratelimit-model-requests-remaining", "N/A"),
            }
            
            # 提取响应内容
            if response.status_code == 200:
                data = response.json()
                response_text = data["choices"][0]["message"]["content"]
                result["success"] = True
                result["response_text"] = response_text
                result["rate_limit_info"] = rate_limit_info
                
                with print_lock:
                    print(f"[{key_index}/{total_keys}] ✅ Key {key_prefix} 测试成功")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                result["rate_limit_info"] = rate_limit_info
                
                with print_lock:
                    print(f"[{key_index}/{total_keys}] ❌ Key {key_prefix} 返回错误: {response.status_code}")
                
    except Exception as e:
        result["error"] = str(e)
        with print_lock:
            print(f"[{key_index}/{total_keys}] ❌ Key {key_prefix} 调用失败: {e}")
    
    return result


def print_summary(results: list):
    """打印测试摘要"""
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    failed_count = total - success_count
    
    print(f"\n总计: {total} 个 API keys")
    print(f"成功: {success_count} 个 | 失败: {failed_count} 个")
    print("-" * 80)
    
    # 打印每个key的详细信息
    for result in results:
        idx = result["key_index"]
        prefix = result["key_prefix"]
        
        print(f"\n[Key #{idx}] {prefix}")
        print("-" * 60)
        
        if result["success"]:
            print(f"  状态: ✅ 成功")
            print(f"  回答: {result['response_text'][:100]}...")
            
            if result["rate_limit_info"]:
                print(f"  限流信息:")
                for key, value in result["rate_limit_info"].items():
                    print(f"    - {key}: {value}")
                
                # 计算剩余百分比
                try:
                    model_remaining = int(result["rate_limit_info"]["模型剩余额度"])
                    model_limit = int(result["rate_limit_info"]["模型当天限额"])
                    remaining_percent = (model_remaining / model_limit) * 100
                    
                    status_icon = "🟢" if remaining_percent > 50 else "🟡" if remaining_percent > 25 else "🔴"
                    print(f"  {status_icon} 模型剩余: {model_remaining}/{model_limit} ({remaining_percent:.1f}%)")
                except (ValueError, ZeroDivisionError, KeyError):
                    pass
        else:
            print(f"  状态: ❌ 失败")
            print(f"  错误: {result['error']}")
            if result["rate_limit_info"]:
                print(f"  限流信息: {result['rate_limit_info']}")


def main():
    start_time = datetime.now()
    
    # 获取一张测试图片
    images = [p for p in BASE_DIR.iterdir() 
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    test_image = images[0]
    
    print("=" * 80)
    print("🚀 多线程 API Key 限流测试")
    print("=" * 80)
    print(f"�️  测试图片: {test_image.name}")
    print(f"🔑 API Keys 数量: {len(API_KEYS)}")
    print(f"🤖 模型: {MODEL_ID}")
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    results = []
    
    # 使用线程池并发测试所有 keys
    with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        # 提交所有任务
        future_to_key = {
            executor.submit(
                test_single_key, 
                api_key, 
                idx + 1, 
                len(API_KEYS), 
                QUESTION, 
                test_image
            ): idx
            for idx, api_key in enumerate(API_KEYS)
        }
        
        # 收集结果
        for future in as_completed(future_to_key):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = future_to_key[future]
                with print_lock:
                    print(f"❌ 线程 #{idx + 1} 执行异常: {e}")
    
    # 按 key_index 排序
    results.sort(key=lambda x: x["key_index"])
    
    # 打印汇总报告
    print_summary(results)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print(f"✅ 所有测试完成！耗时: {duration:.2f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
