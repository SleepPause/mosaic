#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：检查 Gemini Balance 返回的响应对象结构
用于诊断为什么无法获取 thinking 内容
"""
import base64
from pathlib import Path
from openai import OpenAI
import json

# 配置
BALANCE_BASE_URL = "http://localhost:8001/hf/v1"
BALANCE_API_KEY = "123456"
MODEL = "gemini-2.5-flash"

# 测试图片路径（使用项目中的任意一张图片）
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"
test_images = list(PIC_DIR.glob("*.jpg"))

if not test_images:
    print("❌ 未找到测试图片")
    exit(1)

test_image = test_images[0]
print(f"🖼️ 使用测试图片: {test_image.name}\n")

# 创建客户端
client = OpenAI(
    base_url=BALANCE_BASE_URL,
    api_key=BALANCE_API_KEY
)

# 读取图片并转换为 base64
with open(test_image, 'rb') as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

# 发送测试请求
print("📤 发送测试请求...\n")
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片中的内容。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.0
)

print("="*80)
print("🔍 响应对象详细信息")
print("="*80)

# 1. 打印完整响应对象（美化格式）
print("\n1️⃣ 完整响应对象 (model_dump):")
print("-"*80)
try:
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"无法序列化: {e}")
    print(response)

# 2. 打印 message 对象的所有属性
print("\n\n2️⃣ message 对象的所有属性 (dir):")
print("-"*80)
message = response.choices[0].message
print(dir(message))

# 3. 打印 message.__dict__
print("\n\n3️⃣ message.__dict__:")
print("-"*80)
print(json.dumps(message.__dict__, indent=2, ensure_ascii=False, default=str))

# 4. 检查可能的 thinking 字段
print("\n\n4️⃣ 检查可能的 thinking 字段:")
print("-"*80)
possible_fields = [
    'reasoning_content',
    'reasoning', 
    'thinking',
    'thought',
    'thinking_process',
    'thoughts',
    'reason'
]

for field in possible_fields:
    value = getattr(message, field, None)
    if value:
        print(f"✅ 字段 '{field}' 存在，值: {value[:100]}..." if len(str(value)) > 100 else f"✅ 字段 '{field}' 存在，值: {value}")
    else:
        print(f"❌ 字段 '{field}' 不存在或为空")

# 5. 打印 content
print("\n\n5️⃣ content 内容:")
print("-"*80)
print(message.content[:500] if message.content and len(message.content) > 500 else message.content)

print("\n\n" + "="*80)
print("✅ 测试完成")
print("="*80)
