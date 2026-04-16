from Agent import Agent
from LLM import LLM
from Message import Message
from Config import Config
from typing import Optional
from Tool import ToolRegistry

class SimpleAgent(Agent):
    def __init__(
            self,
            name: str,
            llm: LLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"{name} 初始化完成，工具调用：{'启用' if enable_tool_calling else '禁用'}")

    def run(self, user_input: str, max_iterations: int = 3, **kwargs) -> str:
        print(f"{self.name}开始处理{user_input}")
        messages = []
        new_prompt = self._update_system_prompt()
        messages.append({"role": "system", "content": new_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_input})

        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message("user", user_input))
            self.add_message(Message("assistant", response))
            return response

        return self._run_tool_calling(messages, max_iterations, **kwargs)

    def _update_system_prompt(self,):
        base_prompt = self.system_prompt if self.system_prompt else "You are a helpful assistant"

        if not self.tool_registry or not self.enable_tool_calling:
            return base_prompt

        tools_descriptions = self.tool_registry.get_tools_descriptions()
        if not tools_descriptions or tools_descriptions == "No tools registered":
            return base_prompt
        
        tools_prompt = """
            你可以使用以下工具:
            {tools}

            当你需要调用工具时，必须严格遵循以下格式：
            '[TOOL_CALL:{{tool_name}}:{{tool_parameters}}]'
            例如：
            '[TOOL_CALL:search:Python编程]' 或 '[TOOL_CALL:memory:recall=用户信息]'

            工具调用结果会插入到对话中，然后你可以基于结果继续回答
        """
        return base_prompt + tools_prompt.format(tools=tools_descriptions)
