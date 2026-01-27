from openai import OpenAI
import time

def main():
    base_url = "http://localhost:4001/v1"
    api_key = "123456"
    model = "gemini-2.5-pro"
    
    print(f"[*] Connecting to: {base_url}")
    print(f"[*] Model: {model}")
    
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print("\n[*] Sending request...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        
        end_time = time.time()
        print(f"\n[+] Success ({end_time - start_time:.2f}s):")
        print("-" * 30)
        print(response.choices[0].message.content)
        print("-" * 30)
        
    except Exception as e:
        print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    main()
