from abc import ABC, abstractmethod
from typing import Optional, List
from LLM import LLM
from Message import Message
from Config import Config


class Agent(ABC):
    def __init__(self, name: str, llm: LLM, system_prompt: Optional[str] = None, config: Optional[Config] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, user_input: str, **kwargs) -> str:
        """运行智能体"""
        pass

    def add_message(self, message: Message):
        self._history.append(message)
    
    def get_history(self) -> List[Message]:
        return self._history.copy()
    
    def clear_history(self):
        self.history.clear()

    def __str__(self) -> str:
        return f"Agent(name: {self.name}, provider: {self.llm.provider})"
