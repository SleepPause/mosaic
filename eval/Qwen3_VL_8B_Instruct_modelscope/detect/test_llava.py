import ollama
from pathlib import Path
import time

# ================= 配置区域 =================
# 1. 设置服务器地址 (因为建立了 SSH 隧道，所以填本地地址)
OLLAMA_HOST = "http://127.0.0.1:11435"

# 2. 指定模型 (必须是服务器上 ollama list 里有的名字)
MODEL_ID = "llava:13b"

# 3. 你的测试图片路径 (这是你本地电脑上的图片路径，不用传到服务器！)
IMAGE_PATH = r"D:\SleepPause\Program\python\mosaic\eval\Qwen3_VL_8B_Instruct_modelscope\detect\testpic.jpg"

# 4. 测试问题
PROMPT = "Describe this image in detail."
# ===========================================

def run_inference():
    # 初始化客户端，指向我们的 SSH 隧道端口
    client = ollama.Client(host=OLLAMA_HOST)
    
    print(f"[*] Connecting to {MODEL_ID} at {OLLAMA_HOST}...")
    
    try:
        start_time = time.time()
        
        # 发送请求
        response = client.chat(
            model=MODEL_ID,
            messages=[{
                'role': 'user',
                'content': PROMPT,
                'images': [IMAGE_PATH]  # 库会自动把本地图片转成 base64 发给服务器
            }],
            options={
                'temperature': 0.2, # 0.0 - 1.0，越低越严谨
                # LLaVA 不需要特意改 num_ctx，它通常自动处理得很好，但也可以显式指定
                # 'num_ctx': 4096 
            }
        )
        
        end_time = time.time()
        
        # 获取结果
        content = response['message']['content']
        
        print("\n" + "="*40)
        print(f"⏱️  耗时: {end_time - start_time:.2f}s")
        print("🤖 LLaVA 回答:")
        print("-" * 40)
        print(content)
        print("="*40 + "\n")
        
        return content

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        # 常见错误提示：
        if "Connection refused" in str(e):
            print("👉 检查 SSH 隧道是否断开？(ssh -L ...)")
        elif "model" in str(e) and "not found" in str(e):
            print("👉 服务器上没有下载这个模型，请在服务器运行 `ollama pull llava:13b`")

if __name__ == "__main__":
    # 简单的单张测试
    if Path(IMAGE_PATH).exists():
        run_inference()
    else:
        print(f"❌ 找不到测试图片: {IMAGE_PATH}")