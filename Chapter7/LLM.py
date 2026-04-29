import os
import dotenv
from openai import OpenAI
from typing import Optional, List, Dict

dotenv.load_dotenv()

class LLM:
    def __init__(
            self, 
            model: str = None, 
            apiKey: str = None, 
            baseUrl: str = None, 
            provider: Optional[str] = "auto",
            **kwargs
    ):
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            
            # 解析 ModelScope 的凭证
            self.api_key = apiKey or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = baseUrl or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            # 默认使用 DeepSeek 模型
            self.model = model or "deepseek-v4-pro"
            self.api_key = apiKey or os.getenv("DS_API_KEY")
            self.base_url = baseUrl or os.getenv("DS_BASE_URL")
            self.timeout = kwargs.get('timeout', 60)

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)


    def stream_invoke(self, messages: List[Dict[str, str]], temperature: float = 0.7):
        """流式调用大语言模型"""
        print(f"==========正在调用大语言模型 {self.model}......==========")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            print("\n==========大模型调用结束==========")
        except Exception as e:
            print(f"调用大模型时发生错误：{e}")
            return ""
        
    def invoke(self, messages: List[Dict[str, str]], **kwargs):
        """非流式调用大语言模型"""
        print(f"==========正在调用大语言模型 {self.model}......==========")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用大模型时发生错误：{e}")
            return ""

if __name__ == "__main__":
    llm = LLM(
        model = "deepseek-r1:14b",
        apiKey = "ollama",
        baseUrl = "http://localhost:11434/v1",
        provider = "ollama"
    )
    for chunk in llm.stream_invoke([{"role": "user", "content": "请推荐三个上海旅游景点"}]):
        pass