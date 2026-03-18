import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

class LLMClient:
    """封装LLM客户端，调用兼容OpenAI API的大模型服务，默认使用流式响应"""
    def __init__(self, model: str = "deepseek-chat", prompt: str = "", context: str = ""):
        self.client = OpenAI(api_key = os.getenv("DS_API_KEY"), base_url = os.getenv("DS_BASE_URL"))
        self.model = model
        self.messages = []

        system_prompt = prompt if prompt else "You are a helpful asistant."
        if context:
            system_prompt += f"\nContext: {context}"
        
        self.messages.append({"role": "system", "content": system_prompt})
    
    def think(self, prompt: str, temperature: float = 0.7):
        """调用大模型进行思考并返回其响应"""
        print(f"==========正在调用{self.model}模型......==========")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = self.messages,
                temperature = temperature,
                stream = True
            )

            answer = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    answer += content
            print("\n==========大模型调用结束==========")
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"调用大模型时发生错误：{e}")
            return ""
    