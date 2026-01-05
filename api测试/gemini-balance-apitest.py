import sys
from openai import OpenAI

# --- 配置部分 ---
# 如果你在服务器本机运行，用 localhost。
# 如果你在自己电脑运行连服务器，把 localhost 改成服务器公网IP。
BASE_URL = "http://localhost:8001/openai/v1"

# 你的 ALLOWED_TOKENS
API_KEY = "123456"

# 你想使用的模型名称
MODEL_NAME = "gemini-2.5-flash"

def main():
    # 1. 初始化客户端
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    # 2. 初始化对话历史 (加入 System Prompt)
    history = [
        {"role": "system", "content": "你是一个智能助手，请用简洁的语言回答问题。"}
    ]

    print(f"🚀 已连接到 Gemini Balance ({BASE_URL})")
    print("💡 输入 'quit' 或 'exit' 结束对话。\n")

    # 3. 进入对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\nYou: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ["quit", "exit"]:
                print("Bye! 👋")
                break
            
            if not user_input:
                continue

            # 将用户问题加入历史
            history.append({"role": "user", "content": user_input})

            # 发送请求 (开启流式 stream=True)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                stream=True
            )

            print("AI: ", end="", flush=True)
            
            # 收集完整的回复内容
            full_reply = ""
            
            # 逐字打印回复
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_reply += content
            
            print() # 换行

            # 将 AI 的回复加入历史，这样下次它就记得了
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            # 处理 Ctrl+C
            print("\n\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()