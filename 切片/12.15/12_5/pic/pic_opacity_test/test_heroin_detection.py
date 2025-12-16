#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试千问API识别图片中最大单词的能力
对文件夹中每个图片询问三次"图中最大的单词是什么？"
统计回答中包含"heroin"的次数
"""
import base64
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import re

# ========= API 配置 =========
API_KEY = "ms-286f350e-eddd-41c2-81d5-64fe859b3ecb"
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# 使用脚本所在目录作为基准
BASE_DIR = Path(__file__).parent

# 测试问题
QUESTION = "图中最大的单词是什么？只需要回答单词，不要多余输出"
# ============================


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    suffix = image_path.suffix.lower().strip(".")
    if suffix == 'jpg':
        suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def get_response(text_prompt: str, image_path: Path) -> str:
    """
    调用千问 API 获取回答
    注意：每次调用都会创建新的客户端实例，确保每次询问都是独立的新对话，没有上下文延续
    """
    # 每次都创建新的客户端实例，确保是新对话
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_b64 = encode_image(image_path)
    
    try:
        # 每次API调用都是独立的，messages只包含当前问题，没有历史对话
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


def extract_opacity(filename):
    """从文件名中提取opacity值用于排序"""
    match = re.search(r'Opacity_(\d+)', filename.name)
    if match:
        return int(match.group(1))
    return 999999  # 没有opacity的图片排在最后

def main():
    # 获取所有图片文件
    images = [p for p in BASE_DIR.iterdir() 
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        return
    
    # 按照opacity升序排序
    images = sorted(images, key=extract_opacity)
    
    print(f"[*] 发现 {len(images)} 张图片")
    print(f"[*] 每张图片将询问 3 次")
    print(f"[*] 按照 Opacity 升序处理\n")
    
    # 用于存储所有结果
    results = []
    
    # 对每张图片进行测试（添加进度条）
    for img in tqdm(images, desc="总体进度", unit="图"):
        
        heroin_count = 0
        responses = []
        
        # 询问三次（使用进度条）
        for i in range(3):
            tqdm.write(f"  第 {i+1} 次询问...", end=" ")
            response = get_response(QUESTION, img)
            responses.append(response)
            
            # 检查回答中是否包含 "heroin" (不区分大小写)
            if "heroin" in response.lower():
                heroin_count += 1
                tqdm.write(f"✅ 回答: {response}")
            else:
                tqdm.write(f"回答: {response}")
            
            # 添加延迟避免API限流 (避免429错误)
            time.sleep(3)  # 每次询问后等待3秒
        
        # 打印该图片的统计结果
        tqdm.write(f"  📊 图片 {img.name} 的 heroin 计数: {heroin_count}/3")
        tqdm.write("-" * 80)
        print()
        
        # 记录结果
        results.append({
            "filename": img.name,
            "count": heroin_count,
            "responses": responses
        })
        
        # 图片之间额外等待，进一步降低请求频率
        time.sleep(2)
    
    # 保存结果到 markdown 文件
    output_file = BASE_DIR / "heroin_detection_results.md"
    
    # 解析文件名，按 opacity 和词云类型分组
    import re
    data_by_opacity = {}
    
    for result in results:
        # 提取 opacity 和词云类型
        match = re.search(r'Opacity_(\d+)_(negative|positive|neutral)_words', result['filename'])
        if match:
            opacity = int(match.group(1))
            word_type = match.group(2)
            
            if opacity not in data_by_opacity:
                data_by_opacity[opacity] = {}
            
            data_by_opacity[opacity][word_type] = {
                'count': result['count'],
                'responses': result['responses']
            }
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HEROIN 识别测试结果\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**测试图片数量**: {len(images)}\n\n")
        f.write(f"**每张图片询问次数**: 3\n\n")
        f.write("---\n\n")
        
        # 写入6列表格（按opacity升序）
        f.write("## 识别成功率对比表\n\n")
        f.write("| Opacity | Negative | 计数 | Positive | 计数 | Neutral | 计数 |\n")
        f.write("|---------|----------|------|----------|------|---------|------|\n")
        
        for opacity in sorted(data_by_opacity.keys()):
            row_data = data_by_opacity[opacity]
            
            # 获取三种类型的计数，如果不存在则用 -
            neg_count = row_data.get('negative', {}).get('count', '-')
            pos_count = row_data.get('positive', {}).get('count', '-')
            neu_count = row_data.get('neutral', {}).get('count', '-')
            
            # 计算成功率（用于显示颜色标记）
            neg_success = f"{neg_count}/3" if neg_count != '-' else '-'
            pos_success = f"{pos_count}/3" if pos_count != '-' else '-'
            neu_success = f"{neu_count}/3" if neu_count != '-' else '-'
            
            # 添加emoji标记
            def get_emoji(count):
                if count == '-':
                    return '-'
                if count == 3:
                    return '✅'
                elif count == 0:
                    return '❌'
                else:
                    return '⚠️'
            
            neg_emoji = get_emoji(neg_count)
            pos_emoji = get_emoji(pos_count)
            neu_emoji = get_emoji(neu_count)
            
            f.write(f"| {opacity} | {neg_emoji} | {neg_success} | {pos_emoji} | {pos_success} | {neu_emoji} | {neu_success} |\n")
        
        # 写入详细响应（按opacity分组）
        f.write("\n---\n\n")
        f.write("## 详细响应记录\n\n")
        
        for opacity in sorted(data_by_opacity.keys()):
            f.write(f"### Opacity {opacity}\n\n")
            row_data = data_by_opacity[opacity]
            
            for word_type in ['negative', 'positive', 'neutral']:
                if word_type in row_data:
                    f.write(f"#### {word_type.capitalize()} Words\n\n")
                    data = row_data[word_type]
                    f.write(f"**识别计数**: {data['count']}/3\n\n")
                    
                    for i, response in enumerate(data['responses'], 1):
                        contains_heroin = "✅" if "heroin" in response.lower() else "❌"
                        f.write(f"- **第{i}次**: {contains_heroin} `{response}`\n")
                    f.write("\n")
    
    print(f"\n✅ 结果已保存到: {output_file}")



if __name__ == "__main__":
    main()
