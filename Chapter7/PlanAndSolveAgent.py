PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串，仅输出该列表。

输出示例：
[子任务1，子任务2，子任务3]
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

import json
from LLM import LLM
from Agent import Agent
from Config import Config
from Message import Message
from Tool import ToolRegistry
from typing import Optional, List, Dict, Any

class Planner:
    def __init__(self, llm: LLM, prompt: Optional[str] = None):
        self.llm = llm
        self.prompt = prompt or PLANNER_PROMPT_TEMPLATE
    
    def plan(self, task: str, **kwargs) -> list[str]:
        print(f"开始规划任务: {task}")
        messages = ({"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"请为任务{task}生成一个行动计划"})
        response = self.llm.invoke(messages, **kwargs)

        try:
            plan = json.loads(response)
            if not isinstance(plan, list):
                raise ValueError("计划必须是一个Python列表")
            
            print(f"生成计划：")
            for i, step in enumerate(plan):
                print(f"{i+1}. {step}")

            return plan
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始输出: {response}")
            return None


class PlanAndSolveAgent(Agent):
    def __init__(
            self,
            name: str,
            llm: LLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            executor_prompt: Optional[str] = None,
            tool_registry: Optional[ToolRegistry] = None,
            enable_tool_calling: bool = True,
            max_iterations: int = 3
        ):
        super().__init__(
            name,
            llm,
            system_prompt = system_prompt or "You are a helpful assistant.",
            config = config,
            tool_registry = tool_registry
        )
        self.executor_prompt = executor_prompt or EXECUTOR_PROMPT_TEMPLATE
        self.enable_tool_calling = enable_tool_calling and self.tool_registry is not None
        self.max_iterations = max_iterations

        self.planner = Planner(llm)

    def run(self, task: str, **kwargs):
        print(f"{self.name} 开始处理任务: {task}")

        plan = self.planner.plan(task, **kwargs)
        if not plan:
            print("无法生成有效的行动计划。")

            self.add_message(Message("user", task))
            self.add_message(Message("assistant", "无法生成有效的行动计划。"))
            return
        
        final_result = self._execute(task, plan, **kwargs)
        print(f"任务执行完成，最终结果: {final_result}")

        self.add_message(Message("user", task))
        self.add_message(Message("assistant", final_result))

        return final_result
    
    def _execute(self, task: str, plan: list[str], **kwargs):
        print(f"开始按计划逐步执行任务: {task}")
        history = []
        final_result = ""

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")
            execute_prompt = self.executor_prompt.format(
                question = task,
                plan = plan,
                history = history or "",
                current_step = step
            )
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": execute_prompt}]
            response = self._execute_step(messages, **kwargs)
            history.append({"step": step, "result": response})
            print(f"步骤 {i+1} 执行结果: {response}")
            final_result = response
        
        return final_result


    def _execute_step(self, messages: List[Dict[str, Any]], **kwargs):
        if not self.enable_tool_calling:
            return self.llm.invoke(messages, **kwargs)
        
        tool_schemas = self._build_tool_schemas()
        for i in range(self.max_iterations):
            print(f"第{i+1}次工具调用")
            try:
                response = self.llm.invoke_with_tools(messages, tool_schemas, **kwargs)
            except Exception as e:
                print(f"工具调用失败: {e}")
                break

            tool_calls = response.tool_calls
            if not tool_calls:
                return response.content or ""
            
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

                print(f"调用工具 {tool_call['name']}，参数: {arguments}")
                result = self._execute_tool(tool_call["name"], arguments)
                print(f"工具 {tool_call['name']} 执行结果: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"工具 {tool_call['name']} 执行结果: {result}"
                })
        
        return self.llm.invoke
