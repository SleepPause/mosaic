#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理 eval_text 目录中的 txt 文件
将大于 50KB 的文件中的 "Model's Response" 字段内容截断到前 300 个字
其他字段保持不变
"""
import os
import re
from pathlib import Path


# ========= 用户配置区域 =========
# 使用脚本所在目录作为基准
BASE_DIR = Path(__file__).parent
EVAL_TEXT_DIR = BASE_DIR / "eval_text1"

# 文件大小阈值（字节）
SIZE_THRESHOLD = 50 * 1024  # 50KB

# 截断字符数
MAX_RESPONSE_LENGTH = 2000
# =================================


def truncate_model_response(content: str, max_length: int = 300) -> str:
    """
    截断文件中的 "Model's Response" 字段内容到指定长度
    
    正则匹配 "Model's Response:" 和 "---\\n### Evaluation Criteria (StrongREJECT Rubric)" 之间的内容
    只保留前 max_length 个字符
    
    Args:
        content: 原始文件内容
        max_length: Model's Response 字段保留的最大字符数
        
    Returns:
        处理后的文件内容
    """
    # 使用正则表达式匹配两个字符串之间的内容
    # Pattern: "Model's Response:" 后面的内容,直到 "---\n### Evaluation Criteria (StrongREJECT Rubric)"
    pattern = r"(Model's Response:\s*)(.*?)(\n---\s*\n### Evaluation Criteria \(StrongREJECT Rubric\))"
    
    def replace_response(match):
        prefix = match.group(1)  # "Model's Response: "
        response_content = match.group(2)  # 实际的响应内容
        suffix = match.group(3)  # "\n---\n### Evaluation Criteria (StrongREJECT Rubric)"
        
        # 如果响应内容超过指定长度，进行截断
        if len(response_content) > max_length:
            truncated = response_content[:max_length]
            return f"{prefix}{truncated}... {suffix}"
        else:
            return match.group(0)  # 不需要截断，返回原文
    
    # 执行替换
    new_content = re.sub(pattern, replace_response, content, flags=re.DOTALL)
    
    return new_content


def remove_thinking_process_truncated(content: str) -> str:
    """
    删除 "... [Thinking Process Truncated]" 字符串
    """
    return content.replace("... [Thinking Process Truncated]", "")


def clean_json_from_thinking_process(content: str) -> str:
    """
    清除 Thinking Process 部分中的 JSON 代码块
    """
    # 匹配 ```json 开始，直到下一个 ** 或文件末尾
    # ``` : 必须有反引号
    # \s*json : json 关键字
    # [\s\S]*? : 非贪婪匹配内容
    # (?=\s*\*\*|$) : 结束条件：下一个 ** (也就是下一个标题) 或 文件末尾
    pattern = r"```\s*json[\s\S]*?(?=\s*\*\*|$)"

    current_content = content
    
    # 循环清理，直到所有匹配项被移除
    while True:
        original_len = len(current_content)
        current_content = re.sub(pattern, "", current_content, flags=re.IGNORECASE)
        if len(current_content) == original_len:
            break
            
    return current_content


def process_file(file_path: Path) -> bool:
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否成功处理
    """
    try:
        # 检查文件大小
        file_size = file_path.stat().st_size
        
        # 读取文件内容
        original_content = file_path.read_text(encoding='utf-8')
        new_content = original_content
        
        # 1. 如果文件大于 50KB，执行截断操作
        if file_size > SIZE_THRESHOLD:
            new_content = truncate_model_response(new_content, MAX_RESPONSE_LENGTH)
            
        # 2. 对所有文件执行去除字符串操作
        new_content = remove_thinking_process_truncated(new_content)
        
        # 3. 如果文件名包含 "JSON" (忽略大小写)，清理 Thinking Process 中的 JSON
        if "JSON" in file_path.name.upper():
            marker = "=== Thinking Process ==="
            parts = new_content.split(marker)
            
            if len(parts) >= 2:
                before_thinking = parts[0]
                thinking_process_content = marker.join(parts[1:]) 
                
                cleaned_thinking = clean_json_from_thinking_process(thinking_process_content)
                new_content = before_thinking + marker + cleaned_thinking
        
        # 检查是否有变化
        if new_content == original_content:
            if file_size <= SIZE_THRESHOLD:
                print(f"⏭️  跳过: {file_path.name} ({file_size / 1024:.2f} KB < 50 KB, 无需清理)")
            else:
                print(f"ℹ️  无需修改: {file_path.name}")
            return False
        
        print(f"📄 处理: {file_path.name} ({file_size / 1024:.2f} KB)")
        
        # 写回文件
        file_path.write_text(new_content, encoding='utf-8')
        
        new_size = file_path.stat().st_size
        saved_bytes = file_size - new_size
        
        print(f"   ✅ 完成！文件大小: {file_size / 1024:.2f} KB → {new_size / 1024:.2f} KB (变化 {saved_bytes / 1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False


def main():
    """主函数"""
    if not EVAL_TEXT_DIR.exists():
        print(f"❌ 目录不存在：{EVAL_TEXT_DIR}")
        return
    
    # 获取所有 txt 文件
    txt_files = list(EVAL_TEXT_DIR.rglob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  未找到任何 txt 文件在 {EVAL_TEXT_DIR}")
        return
    
    print(f"{'='*70}")
    print(f"批量处理 eval_text 目录中的 txt 文件")
    print(f"{'='*70}")
    print(f"目录: {EVAL_TEXT_DIR}")
    print(f"文件总数: {len(txt_files)}")
    print(f"大小阈值: {SIZE_THRESHOLD / 1024:.0f} KB")
    print(f"Response 最大长度: {MAX_RESPONSE_LENGTH} 字符")
    print(f"{'='*70}\n")
    
    # 统计
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个文件
    for txt_file in txt_files:
        file_size = txt_file.stat().st_size
        result = process_file(txt_file)
        
        if result:
            processed_count += 1
        else:
            skipped_count += 1
    
    # 输出统计信息
    print(f"\n{'='*70}")
    print(f"处理完成")
    print(f"{'='*70}")
    print(f"✅ 成功处理: {processed_count} 个文件")
    print(f"⏭️  跳过（小于 50KB）: {skipped_count} 个文件")
    print(f"❌ 处理失败: {error_count} 个文件")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
