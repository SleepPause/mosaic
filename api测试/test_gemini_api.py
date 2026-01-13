#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini API 简单测试脚本
用于测试特定 API Key 与特定模型的交互
"""
from google import genai
from google.genai import types

# ========= 配置区域 =========
# 在这里写死 API Key
API_KEY = "AIzaSyCNxzyjMWHrB0Z66Th49QDpChiNvn11ByE"  # 请替换为你的实际 API Key

# 模型选择
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"  # 可选: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash-exp 等

# 测试提示词
TEST_PROMPT = "请用中文介绍一下你自己，并说明你能做什么。"
# ================================


def test_gemini_api():
    """测试 Gemini API 调用"""
    print("=" * 60)
    print("Gemini API 测试脚本")
    print("=" * 60)
    print(f"使用模型: {MODEL_NAME}")
    print(f"测试提示: {TEST_PROMPT}\n")
    print("-" * 60)
    
    try:
        # 创建客户端
        print("🔧 正在初始化 Gemini 客户端...")
        client = genai.Client(api_key=API_KEY)
        print("✅ 客户端初始化成功\n")
        
        # 配置 thinking（思考过程）
        thinking_config = types.ThinkingConfig(include_thoughts=True)
        
        if "gemini-2.5" in MODEL_NAME:
            thinking_config.thinking_budget = 8192  # medium 级别
        elif "gemini-3" in MODEL_NAME:
            thinking_config.thinking_level = types.ThinkingLevel.MEDIUM
        
        # 调用 API（流式响应）
        print("📡 正在调用 Gemini API...")
        print("=" * 60)
        
        response = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=TEST_PROMPT,
            config=types.GenerateContentConfig(
                temperature=0.7,
                thinking_config=thinking_config
            )
        )
        
        answer_content = ""
        reasoning_content = ""
        
        # 处理流式响应
        for chunk in response:
            if not chunk.candidates:
                continue
            
            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                
                # 判断是思考内容还是回答内容
                if part.thought:  # 思考内容
                    reasoning_content += part.text
                else:  # 回答内容
                    answer_content += part.text
                    # 实时打印回答内容
                    print(part.text, end="", flush=True)
        
        print("\n" + "=" * 60)
        
        # 输出思考过程（如果有）
        if reasoning_content.strip():
            print("\n💭 思考过程:")
            print("-" * 60)
            print(reasoning_content.strip())
            print("-" * 60)
        
        # 输出统计信息
        print(f"\n📊 统计信息:")
        print(f"   - 回答长度: {len(answer_content)} 字符")
        print(f"   - 思考过程长度: {len(reasoning_content)} 字符")
        print("\n✅ 测试成功完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini_api()
