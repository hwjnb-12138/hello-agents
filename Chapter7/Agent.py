from abc import ABC, abstractmethod
from typing import Optional, List
from LLM import LLM
from Message import Message
from Config import Config
from Tool import ToolRegistry, Tool
from typing import Dict, Any


class Agent(ABC):
    def __init__(
            self, 
            name: str, 
            llm: LLM, 
            system_prompt: Optional[str] = None, 
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None
        ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []
        self.tool_registry = tool_registry

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

    def list_tools(self):
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def add_tool(self, name: str, description: str,tool: Optional[Tool] = None, func: Optional[callable] = None):
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()

        if tool:
            self.tool_registry.register_tool(tool)

        if func:
            self.tool_registry.register_function(name, description, func)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            self.tool_registry.unregister_tool(tool_name)
            return True
        return False

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具"""
        if not self.tool_registry:
            return "未配置工具注册表，无法调用工具"
        
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                response = tool.run(arguments)
                return response
            except Exception as e:
                return f"调用工具 {tool_name} 失败：{e}"
        
        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                response = func(input_text)
                return response
            except Exception as e:
                return f"调用函数 {tool_name} 失败：{e}"
        
        return f"未找到工具或函数 {tool_name}"

    def __str__(self) -> str:
        return f"Agent(name: {self.name}, provider: {self.llm.provider})"
