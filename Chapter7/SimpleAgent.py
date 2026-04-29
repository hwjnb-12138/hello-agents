import re
from typing import Dict, Any, Iterator
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
        super().__init__(name, llm, system_prompt, config, tool_registry)
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"{name} 初始化完成，工具调用：{'启用' if self.enable_tool_calling else '禁用'}")

    def stream_run(self, user_input: str, **kwargs) -> Iterator[str]:
        """流式运行"""
        print(f"{self.name} 开始流式处理 {user_input}")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_input})

        final_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            if chunk is None:
                continue
            final_response += chunk
            print(chunk, end="", flush = True)
            yield chunk
        print()

        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))
        print(f"{self.name} 流式处理完成")


    def run(self, user_input: str, max_iterations: int = 3, **kwargs) -> str:
        print(f"{self.name} 开始处理 {user_input}")

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
            print(f"{self.name} 处理完成")
            return response

        return self._run_tool_calling(messages, user_input, max_iterations, **kwargs)
    
    def _run_tool_calling(self, messages: list, user_input: str, max_iterations: int, **kwargs) -> str:
        current_iteration = 0
        final_response = ""

        while current_iteration < max_iterations:
            current_iteration += 1
            print(f"\n--- 迭代 {current_iteration} ---")
            
            response = self.llm.invoke(messages, **kwargs)
            print(f"[Agent 回复]: {response}")

            tool_calls = self._parse_tool_call(response)
            if not tool_calls:
                final_response = response
                break
            
            tool_results = []
            for tool in tool_calls:
                print(f"[工具调用]: {tool['name']} -> {tool['parameters']}")
                parameters = self._parse_tool_parameters(tool["name"], tool["parameters"])
                result = self._execute_tool(tool["name"], parameters)
                tool_results.append(f"工具{tool['name']}执行结果：{result}")
                print(f"[工具结果]: {result}")
            
            tool_results_text = "\n".join(tool_results)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n请根据这些结果继续处理用户请求"})

        if current_iteration >= max_iterations and not final_response:
            print("[提示] 达到最大迭代次数，获取最终回答")
            final_response = self.llm.invoke(messages, **kwargs)
        
        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_response))
        print(f"\n{self.name} 处理完成，结果：{final_response}")

        return final_response
    
    def _parse_tool_call(self, text: str) -> list:
        """解析需要调用的工具"""

        pattern = r"\[TOOL_CALL:([^:]+):(.*?)\]"
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, tool_parameters in matches:
            tool_calls.append({
                "name": tool_name,
                "parameters": tool_parameters,
                "original": f"[TOOL_CALL:{tool_name}:{tool_parameters}]"
            })

        return tool_calls

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> Dict[str, Any]:
        """解析工具参数"""
        parameters_dict = {}
        
        if "=" in parameters:
            if "," in parameters:
                for param in parameters.split(","):
                    key, value = param.split("=", 1)
                    parameters_dict[key.strip()] = value.strip()
            else:
                key, value = parameters.split("=", 1)
                parameters_dict[key.strip()] = value.strip()
        else:
            if tool_name == "search":
                parameters_dict = {"query": parameters}
            elif tool_name == "memory":
                parameters_dict = {"action": "search", "query": parameters}
            else:
                parameters_dict = {"input": parameters}

        return parameters_dict

    def _update_system_prompt(self):
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
