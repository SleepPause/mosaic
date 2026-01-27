import time
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import ollama
from tqdm import tqdm

# 导入策略库（新版：使用模板函数）
import prompt_strategies_new as prompt_strategies
import evaluation_templates

# ========= 用户配置区域 =========
MODEL_ID = "llava:13b"
BASE_DIR = Path(__file__).parent
PIC_DIR = BASE_DIR.parent / "pic"
OUT_DIR = BASE_DIR / "eval_text"
MAX_WORKERS = 2  # 双卡并发

# 每个策略重复测试的次数
REPEAT_TIMES = 6

EVAL_TEMPLATE = evaluation_templates.EVAL_TEMPLATE_STRONGREJECT
# =================================


def get_strategies_for_target(target_word: str) -> dict:
    """
    根据目标词汇动态生成策略集合
    
    Args:
        target_word: 从图片文件名提取的目标词（如 "HEROIN", "METH"）
    
    Returns:
        策略名称到提示词文本的映射字典
    """
    # 从映射字典中获取对应的 action，如果不存在则使用默认值
    action = prompt_strategies.TARGET_CONTEXTS.get(target_word, "synthesis process")
    
    return {
        "BreakingBad": prompt_strategies.get_breaking_bad_prompt(target_word, action),
        "SafetyAuditor": prompt_strategies.get_safety_auditor_prompt(target_word, action),
        #"Historian": prompt_strategies.get_historian_prompt(target_word, action),
        "JSON": prompt_strategies.get_json_format_prompt(target_word, action),
       # "ArtCritic": prompt_strategies.get_art_critic_prompt(target_word, action),
    }


def extract_target_from_filename(filename: str) -> str:
    """
    从文件名中提取目标词汇
    
    例如: "HEROIN_Opacity_64_negative_words.jpg" -> "HEROIN"
          "METH_test.jpg" -> "METH"
    
    Args:
        filename: 图片文件名（不含路径）
    
    Returns:
        提取的目标词（大写），如果无法提取则返回 "HEROIN" 作为默认值
    """
    # 假设文件名格式为: TARGET_其他信息.jpg
    parts = filename.split('_')
    if parts:
        target = parts[0].upper()
        # 验证是否在已知目标列表中
        if target in prompt_strategies.TARGET_CONTEXTS:
            return target
    
    # 如果无法识别，默认返回 HEROIN
    print(f"⚠️  无法从文件名 '{filename}' 提取目标词，使用默认值 HEROIN")
    return "HEROIN"


def safe_file_name(name: str) -> str:
    """去掉特殊符号"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def process_single_task(task_args):
    """
    工作线程只负责跑，不负责打印。
    它把结果（成功或失败的信息）return 出去，交给主线程打印。
    """
    image_path, target_word, strat_name, prompt_text, idx = task_args
    img_name = image_path.name
    
    # 从图片文件名中提取 method 标记（methodA/methodB）
    img_stem = image_path.stem
    method_suffix = ""
    method_name = ""  # 用于创建目录
    
    # 检查是否包含 _methodX 格式
    if "_method" in img_stem:
        parts = img_stem.split("_")
        for i, part in enumerate(parts):
            if part.startswith("method"):
                method_suffix = f"_{part}"
                method_name = part  # 提取 methodA 或 methodB
                # 重建不包含 method 的文件名
                img_stem = "_".join(parts[:i] + parts[i+1:])
                break
    
    # 创建按 method 和 target 分类的目录结构
    # OUT_DIR/methodX/TARGET/
    if method_name:
        target_dir = OUT_DIR / method_name / target_word
    else:
        # 如果没有 method 标记，使用默认目录
        target_dir = OUT_DIR / target_word
    
    # 确保目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 构造文件名
    file_prefix = f"{safe_file_name(img_stem)}_{strat_name}_{idx}_{MODEL_ID.replace(':','-')}{method_suffix}"
    
    # 跳过已存在的（断点续传）
    existing = list(target_dir.glob(f"{file_prefix}_*.txt"))
    if existing:
        return f"⏭️  [Skip] {img_name} ({target_word}) | {strat_name} - File exists"

    start_t = time.time()
    try:
        # === 调用 Ollama ===
        response = ollama.chat(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": prompt_text,
                "images": [str(image_path)]
            }]
        )
        
        answer_content = response['message']['content']
        duration = time.time() - start_t

        # === 填充模板 ===
        context_snippet = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
        full_content = EVAL_TEMPLATE.replace("[PROMPT_CONTEXT]", context_snippet).replace("[MASK]", answer_content)
        
        full_content += f"\n\n=== Meta Info ===\nModel: {MODEL_ID}\nTime: {duration:.2f}s\nStrategy: {strat_name}\nTarget: {target_word}"

        # === 保存文件 ===
        time_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        file_name = f"{file_prefix}_{time_str}.txt"
        out_path = target_dir / file_name
        out_path.write_text(full_content, encoding='utf-8')

        return f"✅ [Done] {img_name} ({target_word}) | {strat_name} | {duration:.2f}s"

    except Exception as e:
        return f"❌ [Error] {img_name} ({target_word}) | {strat_name}: {str(e)}"


def main():
    if not PIC_DIR.exists():
        print(f"❌ 目录不存在：{PIC_DIR}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [p for p in PIC_DIR.iterdir() if p.suffix.lower() in extensions]
    
    if not images:
        print(f"⚠️  {PIC_DIR} 中没有找到图片")
        return

    # === 生成任务清单（支持多目标词汇）===
    tasks = []
    for img in images:
        # 从文件名提取目标词汇
        target_word = extract_target_from_filename(img.stem)
        # 根据目标词汇动态生成策略
        STRATEGIES = get_strategies_for_target(target_word)
        
        for strat_name, prompt_text in STRATEGIES.items():
            for i in range(1, REPEAT_TIMES + 1):
                tasks.append((img, target_word, strat_name, prompt_text, i))

    print(f"[*] Model: {MODEL_ID}")
    print(f"[*] 目标词汇支持: {', '.join(prompt_strategies.TARGET_CONTEXTS.keys())}")
    print(f"[*] Total Tasks: {len(tasks)} (Images: {len(images)} x Strategies: 3 x Repeat: {REPEAT_TIMES})")
    print(f"[*] Hardware: Dual GPU (Workers: {MAX_WORKERS})")
    print(f"[*] 输出目录: {OUT_DIR}")
    print("-" * 60)

    # === 并发执行 + tqdm 进度条 ===
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. 提交所有任务拿到 futures
        futures = [executor.submit(process_single_task, task) for task in tasks]
        
        # 2. 使用 tqdm 包装 as_completed(futures)
        progress_bar = tqdm(as_completed(futures), total=len(tasks), unit="task", desc="Running RedTeam")
        
        for future in progress_bar:
            result_msg = future.result()
            # 3. 使用 tqdm.write 打印日志，避免打断进度条动画
            tqdm.write(result_msg)

    print("-" * 60)
    print(f"🎉 所有测试完成！")
    print(f"结果已按 method 和 target 分类保存在: {OUT_DIR}")
    print(f"\n目录结构示例:")
    print(f"  {OUT_DIR}/methodA/HEROIN/")
    print(f"  {OUT_DIR}/methodB/COCAINE/")

if __name__ == "__main__":
    main()
