#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试千问 Qwen3-VL-30B-Thinking API 识别图片中最大单词的能力
对文件夹中每个图片询问三次"图中最大的单词是什么？"
统计回答中包含"heroin"的次数，并记录思考过程
"""
import base64
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import re
import sys
import os

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

# ========= API 配置 =========
# 尝试从环境变量获取，如果没有则使用默认值或空字符串
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID = "qwen3-vl-30b-a3b-thinking"

# 图片所在目录 (硬编码为之前的测试目录)
IMAGE_DIR = Path(r"d:\SleepPause\Program\python\mosaic\12_5\pic\pic_opacity_test\pic_opacity")
# 脚本所在目录 (用于保存日志和结果)
BASE_DIR = Path(__file__).parent

# 测试问题
QUESTION = "图中由小单词组成的大单词是什么？输出它，不要输出多余内容"
# ============================


def encode_image(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    suffix = image_path.suffix.lower().strip(".")
    if suffix == 'jpg':
        suffix = 'jpeg'
    mime = f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def get_response(text_prompt: str, image_path: Path) -> tuple[str, str]:
    """
    调用千问 API 获取回答
    返回: (完整回复, 思考过程)
    """
    if not API_KEY:
        return "Error: DASHSCOPE_API_KEY not found in environment variables.", ""

    # 每次都创建新的客户端实例
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_b64 = encode_image(image_path)
    
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
            stream=True,
            extra_body={
                'enable_thinking': True,
                "thinking_budget": 81920
            } if "thinking" in MODEL_ID.lower() else None
        )
        
        for chunk in stream:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # 获取思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            
            # 获取回复内容
            if delta.content:
                answer_content += delta.content
                
        return answer_content.strip(), reasoning_content.strip()
        
    except Exception as e:
        print(f"  ❌ API 调用失败: {e}")
        return f"API Error: {e}", ""


def extract_opacity(filename):
    """从文件名中提取opacity值用于排序"""
    match = re.search(r'Opacity_(\d+)', filename.name)
    if match:
        return int(match.group(1))
    return 999999  # 没有opacity的图片排在最后

def main():
    # 启用日志记录
    log_file = BASE_DIR / f"test_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"[*] 日志文件: {log_file}")
    print(f"[*] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] 图片目录: {IMAGE_DIR}")
    print(f"[*] 模型 ID: {MODEL_ID}\n")
    
    if not API_KEY:
        print("⚠️  警告: 未找到 DASHSCOPE_API_KEY 环境变量。请确保已设置。")
    
    # 获取所有图片文件
    if not IMAGE_DIR.exists():
        print(f"⚠️  图片目录不存在: {IMAGE_DIR}")
        logger.close()
        return

    images = [p for p in IMAGE_DIR.iterdir() 
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    
    if not images:
        print("⚠️  未找到任何图片")
        logger.close()
        return
    
    # 按照opacity升序排序
    images = sorted(images, key=extract_opacity)
    
    print(f"[*] 发现 {len(images)} 张图片")
    print(f"[*] 每张图片将询问 3 次")
    print(f"[*] 按照 Opacity 升序处理\n")
    
    # 用于存储所有结果
    results = []
    
    # 定义结果保存路径
    output_file = BASE_DIR / "heroin_detection_results_thinking.md"
    
    def save_report(current_results):
        """保存当前结果到Markdown文件"""
        # 解析文件名，按 opacity 和词云类型分组
        data_by_opacity = {}
        
        for result in current_results:
            # 提取 opacity 和词云类型
            match = re.search(r'Opacity_(\d+)_(negative|positive|neutral)_words', result['filename'])
            if match:
                opacity = int(match.group(1))
                word_type = match.group(2)
                
                if opacity not in data_by_opacity:
                    data_by_opacity[opacity] = {}
                
                data_by_opacity[opacity][word_type] = {
                    'count': result['count'],
                    'responses': result['responses'],
                    'reasonings': result['reasonings']
                }
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# HEROIN 识别测试结果 (Qwen3-VL-30B-Thinking)\n\n")
            f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**模型**: {MODEL_ID}\n\n")
            f.write(f"**测试图片数量**: {len(images)}\n\n")
            f.write(f"**已完成数量**: {len(current_results)}\n\n")
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
                        
                        for i, (response, reasoning) in enumerate(zip(data['responses'], data['reasonings']), 1):
                            contains_heroin = "✅" if "heroin" in response.lower() else "❌"
                            f.write(f"- **第{i}次**: {contains_heroin} `{response}`\n")
                            if reasoning:
                                f.write(f"  > **思考过程**: {reasoning[:500]}... (已截断)\n")
                        f.write("\n")

    # 对每张图片进行测试（添加进度条）
    for img in tqdm(images, desc="总体进度", unit="图"):
        
        heroin_count = 0
        responses = []
        reasonings = []
        
        # 询问三次（使用进度条）
        for i in range(3):
            tqdm.write(f"  第 {i+1} 次询问...", end=" ")
            response, reasoning = get_response(QUESTION, img)
            responses.append(response)
            reasonings.append(reasoning)
            
            # 检查回答中是否包含 "heroin" (不区分大小写)
            if "heroin" in response.lower():
                heroin_count += 1
                tqdm.write(f"✅ 回答: {response}")
            else:
                tqdm.write(f"回答: {response}")
            
            # 添加延迟避免API限流
            time.sleep(2)
        
        # 打印该图片的统计结果
        tqdm.write(f"  📊 图片 {img.name} 的 heroin 计数: {heroin_count}/3")
        tqdm.write("-" * 80)
        print()
        
        # 记录结果
        results.append({
            "filename": img.name,
            "count": heroin_count,
            "responses": responses,
            "reasonings": reasonings
        })
        
        # 每处理完一张图片就保存一次报告
        try:
            save_report(results)
        except Exception as e:
            tqdm.write(f"⚠️ 保存报告失败: {e}")
        
        # 图片之间额外等待
        time.sleep(1)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"✅ 日志已保存到: {log_file}")
    print(f"[*] 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 关闭日志
    logger.close()
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
