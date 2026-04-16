import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()


class DeepSeek:
    def __init__(self, prompt: str = "", context: str = ""):
        self.client = OpenAI(api_key = os.getenv("DS_API_KEY"), base_url = os.getenv("DS_BASE_URL"))
        self.messages = []

        systemContent = prompt if prompt else "You are a helpful asistant."
        if context:
            systemContent += f"\nContext: {context}" 

        self.messages.append({"role": "system", "content": systemContent})


    def chat(self, prompt: str):
        print("==========正在调用大语言模型......==========")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model = "deepseek-chat",
            messages = self.messages,
        )
        answer = response.choices[0].message.content
        self.messages.append(response.choices[0].message)
        
        return answer