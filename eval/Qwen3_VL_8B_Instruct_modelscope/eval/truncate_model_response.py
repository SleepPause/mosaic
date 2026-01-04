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
EVAL_TEXT_DIR = BASE_DIR / "eval_text"

# 文件大小阈值（字节）
SIZE_THRESHOLD = 50 * 1024  # 50KB

# 截断字符数
MAX_RESPONSE_LENGTH = 1600
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
        
        if file_size <= SIZE_THRESHOLD:
            print(f"⏭️  跳过: {file_path.name} ({file_size / 1024:.2f} KB < 50 KB)")
            return False
        
        print(f"📄 处理: {file_path.name} ({file_size / 1024:.2f} KB)")
        
        # 读取文件内容
        original_content = file_path.read_text(encoding='utf-8')
        
        # 截断 Model's Response
        new_content = truncate_model_response(original_content, MAX_RESPONSE_LENGTH)
        
        # 检查是否有变化
        if new_content == original_content:
            print(f"   ℹ️  无需修改（Model's Response 未超过 {MAX_RESPONSE_LENGTH} 字符）")
            return False
        
        # 写回文件
        file_path.write_text(new_content, encoding='utf-8')
        
        new_size = file_path.stat().st_size
        saved_bytes = file_size - new_size
        
        print(f"   ✅ 完成！文件大小: {file_size / 1024:.2f} KB → {new_size / 1024:.2f} KB (节省 {saved_bytes / 1024:.2f} KB)")
        
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
    txt_files = list(EVAL_TEXT_DIR.glob("*.txt"))
    
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
        elif file_size <= SIZE_THRESHOLD:
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
