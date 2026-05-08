REACT_PROMPT_TEMPLATE = """你是一位具备推理和执行能力的智能助手，你需要根据用户的输入，自主规划并执行动作来帮助用户解决问题

你需要通过连续的思考，决定下一步采取的行动，并且可以通过调用工具来完成任务：
1. **Thought 工具**：用于记录你的推理过程和分析
   - 在需要思考时调用
   - 参数：reasoning（你的推理内容）

2. **业务工具**：用于获取信息或执行操作
   - 根据任务需求选择合适的工具
   - 可以多次调用不同工具

3. **Finish 工具**：用于返回最终答案
   - 当你有足够信息得出结论时调用
   - 参数：answer（最终答案）

你每次只能进行一次Thought、一次执行工具调用或一个Finish
可以根据任务需求，多次调用不同的工具
当你有足够信息来得出结论时，才能调用Finish
"""

import json

from Agent import Agent
from LLM import LLM
from Config import Config
from Tool import ToolRegistry
from Message import Message
from typing import Optional, List, Dict, Any

class ReActAgent(Agent):
    def __init__(
            self,
            name: str,
            llm: LLM,
            prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional[ToolRegistry] = None,
            max_iterations: int = 5
    ):
        super().__init__(
            name,
            llm,
            system_prompt = prompt if prompt else REACT_PROMPT_TEMPLATE,
            config = config,
            tool_registry = tool_registry or ToolRegistry()
        )
        self.max_iterations = max_iterations
    
    def run(self, user_input: str, **kwargs) -> str:
        messages = []
        current_iteration = 0
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})
        tool_schemas = self._build_tool_schemas()

        print(f"\n{self.name} 开始处理任务：{user_input}")
        while current_iteration < self.max_iterations:
            current_iteration += 1
            print(f"\n第 {current_iteration} 次迭代")
            try:
                response = self.llm.invoke_with_tools(
                    messages = messages,
                    tools = tool_schemas,
                    **kwargs
                )
            except Exception as e:
                print(f"LLM 调用失败：{e}")
                break
        
            tool_calls = response.tool_calls
            if not tool_calls:
                final_response = response.content or "抱歉，我无法回答您的问题"
                print(f"\n直接回复：{final_response}")

                self.add_message(Message("user", user_input))
                self.add_message(Message("assistant", final_response))

                return final_response
            
            messages.append({
                "role": "assistant",
                "content": response.content,
                "reasoning_content": response.reasoning_content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    for tc in tool_calls
                ]
            })
            
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call["arguments"])
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"工具调用参数格式错误：{e}"
                    })
                    continue

                if tool_call["name"] == "Thought":
                    reasoning = arguments.get("reasoning", "")
                    print(f"{tool_call['name']}：{reasoning}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"当前思考过程：{reasoning}"
                    })
                elif tool_call["name"] == "Finish":
                    final_response = arguments.get("answer", "")
                    self.add_message(Message("user", user_input))
                    self.add_message(Message("assistant", final_response))
                    print(f"最终结果：{final_response}")

                    return final_response
                else:
                    print(f"调用工具：{tool_call['name']}（{arguments}）")
                    result = self._execute_tool(tool_call["name"], arguments)
                    print(f"工具执行结果：{result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"工具{tool_call['name']}执行结果：{result}"
                    })
        
        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_answer))
        
        return final_answer
        
            
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []

        if self.tool_registry:
            schemas.extend(super()._build_tool_schemas())

        schemas.append({
            "type": "function",
            "function": {
                "name": "Thought",
                "description": "分析问题，制定策略，记录推理过程。在需要思考时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "当前的思考过程与结果"
                        }
                    },
                    "required": ["reasoning"]
                }
            }
        })

        schemas.append({
            "type": "function",
            "function": {
                "name": "Finish",
                "description": "当你有足够信息来得出结论时，调用此工具返回结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "最终结果"
                        }
                    },
                    "required": ["answer"]
                }
            }
        })

        return schemas