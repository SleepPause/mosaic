#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试千问API的响应头，监控剩余额度
通过HTTP响应头来查看API限流信息
"""
import base64
from pathlib import Path
from openai import OpenAI

# ========= API 配置 =========
API_KEY = "ms-286f350e-eddd-41c2-81d5-64fe859b3ecb"
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# 使用脚本所在目录作为基准
BASE_DIR = Path(__file__).parent

# 测试问题
QUESTION = "图中最大的单词是什么？只需要回答单词，不要多于输出"
# ============================


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    suffix = image_path.suffix.lower().strip(".")
    if suffix == 'jpg':
        suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def get_response_with_headers(text_prompt: str, image_path: Path):
    """
    调用千问 API 获取回答，并返回响应头信息
    返回: (response_text, headers_dict)
    """
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_b64 = encode_image(image_path)
    
    try:
        # 使用非流式调用以获取完整响应（包括响应头）
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }],
            stream=False,
        )
        
        # 提取响应内容
        response_text = completion.choices[0].message.content.strip()
        
        # 注意：OpenAI Python SDK 可能不直接暴露响应头
        # 这里我们需要使用底层的httpx客户端来获取响应头
        # 但为了演示，我们先返回基本信息
        
        return response_text, None
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return f"API Error: {e}", None


def get_response_with_httpx(text_prompt: str, image_path: Path):
    """
    使用 httpx 直接调用 API，以获取响应头
    """
    import httpx
    
    image_b64 = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
                return response_text, rate_limit_info
            else:
                return f"Error {response.status_code}: {response.text}", rate_limit_info
                
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return f"API Error: {e}", None


def main():
    # 获取一张测试图片
    images = [p for p in BASE_DIR.iterdir() 
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    test_image = images[0]
    print(f"🖼️  使用测试图片: {test_image.name}\n")
    
    # 测试获取响应头
    print("=" * 80)
    print("正在调用 API 并获取响应头信息...")
    print("=" * 80)
    
    response_text, rate_limit_info = get_response_with_httpx(QUESTION, test_image)
    
    print(f"\n📝 模型回答: {response_text}\n")
    
    if rate_limit_info:
        print("📊 API 限流信息:")
        print("-" * 60)
        for key, value in rate_limit_info.items():
            print(f"  {key}: {value}")
        print("-" * 60)
        
        # 计算剩余百分比
        try:
            model_remaining = int(rate_limit_info["模型剩余额度"])
            model_limit = int(rate_limit_info["模型当天限额"])
            remaining_percent = (model_remaining / model_limit) * 100
            
            print(f"\n⚠️  模型剩余额度: {model_remaining}/{model_limit} ({remaining_percent:.1f}%)")
            
            if remaining_percent < 10:
                print("🚨 警告：模型额度即将用尽！请减慢请求速度。")
            elif remaining_percent < 25:
                print("⚠️  注意：模型额度已使用超过75%")
        except (ValueError, ZeroDivisionError):
            pass
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
