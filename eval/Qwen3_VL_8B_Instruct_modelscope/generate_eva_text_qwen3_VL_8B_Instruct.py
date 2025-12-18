#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量红队评估：遍历 pic/内所有图片，使用多种提示词策略截取模型的回答
文件名：图片名_策略名_YYYYMMDDhhmmss.txt
"""
import time
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import prompt_strategies  # 导入提示词库
import evaluation_templates # 导入评估模板库

# ========= 用户配置区域 =========
API_KEY = ""
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# 使用脚本所在目录作为基准，防止在不同目录下运行时找不到文件
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR / "pic"
OUT_DIR = BASE_DIR / "eval_text"

# 定义要测试的策略集合
# 你可以在这里注释掉不想跑的策略
STRATEGIES = {
    "BreakingBad": prompt_strategies.PROMPT_BREAKING_BAD,
    "SafetyAuditor": prompt_strategies.PROMPT_SAFETY_AUDITOR,
    "Historian": prompt_strategies.PROMPT_HISTORIAN,
    "JSON": prompt_strategies.PROMPT_JSON_FORMAT,
    "ArtCritic": prompt_strategies.PROMPT_ART_CRITIC,
}
# =================================

# 使用 StrongREJECT 模板
EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT


def encode_image(image_path: Path) -> str:
    suffix = image_path.suffix.lower().strip(".")
    # 简单的 mime type 映射
    if suffix == 'jpg': suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def get_response(text_prompt: str, image_path: Path) -> str:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_b64 = encode_image(image_path)
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }],
            stream=True,
        )
        full = ""
        for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full += delta.content
        return full.strip()
    except Exception as e:
        print(f"  ❌ API 调用失败: {e}")
        return f"API Error: {e}"


def safe_file_name(name: str) -> str:
    """去掉特殊符号"""
    import re
    return re.sub(r'[<>:\"/\\|?*]', '_', name)


def main():
    if not PIC_DIR.exists():
        print(f"❌ 目录不存在：{PIC_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = [p for p in PIC_DIR.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return

    print(f"[*] 发现 {len(images)} 张图片，准备测试 {len(STRATEGIES)} 种策略。")
    print(f"[*] 总计请求数: {len(images) * len(STRATEGIES)}")

    for img in images:
        print(f"\n🖼️  正在处理图片: {img.name}")
        
        for strat_name, prompt_text in STRATEGIES.items():
            print(f"  -> 正在测试策略: [{strat_name}] ... ", end="", flush=True)
            
            start_t = time.time()
            answer = get_response(prompt_text, img)
            duration = time.time() - start_t
            
            # 生成文件名: 图片名_策略名_时间戳.txt
            time_str = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{safe_file_name(img.stem)}_{strat_name}_{time_str}.txt"
            out_path = OUT_DIR / file_name
            
            # 填充模板
            # StrongREJECT 模板需要上下文 [PROMPT_CONTEXT]
            # 我们把提示词的前 200 个字符放进去作为上下文，帮助裁判理解意图
            context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            
            content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer)
            out_path.write_text(content, encoding='utf-8')
            
            print(f"✅ 完成 ({duration:.2f}s)")
            
            # 稍微休息一下防止 QPS 超限
            time.sleep(1)

    print("\n🎉 所有测试已完成！结果保存在 eval_text 目录。")


if __name__ == "__main__":
    main()
