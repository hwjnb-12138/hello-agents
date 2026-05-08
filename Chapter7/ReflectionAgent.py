INITIAL_PROMPT_TEMPLATE = """你是一位具有自我反思能力的智能体，会通过多次的反思-优化循环来追求更高质量的解决方案
你的工作流程应该是：
1. 首先尝试根据用户的输入完成相应任务
2. 反思你的回答，找出是否存在可以优化的地方
3. 根据反思结果优化你的回答
4. 如果你认为当前结果已达最优，回复“无需改进”
"""

REFLECT_PROMPT_TEMPLATE = """请仔细审查以下回答，并找出可能的问题或改进空间：
# 原始任务:
{task}

# 当前回答:
{result}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
"""

REFINE_PROMPT_TEMPLATE = """请根据反馈意见改进你的回答：
# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""

import json
from LLM import LLM
from Agent import Agent
from Config import Config
from Message import Message
from Tool import ToolRegistry
from typing import Optional, Dict, Any, List


class Memory:
    def __init__(self):
        self.memories: List[Dict[str, Any]] = []

    def add_memory(self, memory_type: str, content: str):
        """
        参数:
        - type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        memory = {"type": memory_type, "content": content}
        self.memories.append(memory)
        print(f"📝 记忆已更新，新增一条 '{memory_type}' 记录。")
    
    def get_trajectory(self):
        trajectory_parts = []
        for memory in self.memories:
            if memory["type"] == "execution":
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{memory["content"]}")
            elif memory["type"] == "reflection":
                trajectory_parts.append(f"--- 评审员反馈 ---\n{memory["content"]}")
        
        return "\n".join(trajectory_parts)
    
    def get_last_execution(self):
        for memory in reversed(self.memories):
            if memory["type"] == "execution":
                return memory["content"]

        return None


class ReflectionAgent(Agent):
    def __init_(
            self,
            name: str,
            llm: LLM,
            config: Optional[Config] = None,
            tool_registry: Optional[ToolRegistry] = None,
            system_prompt: Optional[str] = None,
            max_reflection_iterations: int = 3,
            max_tool_iterations: int = 3,
            enable_tool_calling: bool = True
    ):
        super().__init__(
            name,
            llm,
            system_prompt = system_prompt or INITIAL_PROMPT_TEMPLATE,
            config = config,
            tool_registry = tool_registry
        )
        self.max_reflection_iterations = max_reflection_iterations
        self.max_tool_iterations = max_tool_iterations
        self.memory = Memory()
        self.enable_tool_calling = enable_tool_calling and self.tool_registry is not None

    def run(self, user_input: str, **kwargs):
        print(f"{self.name} 开始执行，用户输入：{user_input}")
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}]
        
        initial_result = self.invoke_llm(messages, **kwargs)
        self.memory.add_memory("execution", initial_result)

        for i in range(self.max_reflection_iterations):
            print(f"第 {i+1} 轮反思")

            print("正在进行反思...")
            reflection_prompt = REFLECT_PROMPT_TEMPLATE.format(task = user_input, result = self.memory.get_last_execution())
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": reflection_prompt}]
            reflection_response = self.invoke_llm(messages, **kwargs)
            self.memory.add_memory("reflection", reflection_response)

            if "无需改进" in reflection_response:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            print("正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task = user_input,
                last_attempt = self.memory.get_last_execution(),
                feedback = reflection_response
            )
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": refine_prompt}]
            refine_response = self.invoke_llm(messages, **kwargs)
            self.memory.add_memory("execution", refine_response)
        
        final_result = self.memory.get_last_execution()
        print("\n最终结果：", final_result)

        self.add_message(Message("user", user_input))
        self.add_message(Message("assistant", final_result))

        return final_result

    def invoke_llm(self, messages: List[Dict[str, Any]], **kwargs):
        if not self.enable_tool_calling or self.tool_registry is None:
            return self.llm.invoke(messages, **kwargs)
        
        tool_schemas = self._build_tool_schemas()
        for i in range(self.max_tool_iterations):
            print(f"第 {i+1} 轮工具调用")
            try:
                response = self.llm.invoke_with_tools(messages, tool_schemas, **kwargs)
            except Exception as e:
                print(f"LLM 调用工具时出错：{e}")
                break
        
            tool_calls = response.tool_calls
            if not tool_calls:
                return response.content or ""
            
            messages.append({
                "role": "assistant",
                "content": response.content,
                "resoning_content": response.resoning_content,
                "tool_calls": [{
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

                print(f"调用工具：{tool_call['name']}（{arguments}）")
                result = self._execute_tool(tool_call["name"], arguments)
                print(f"工具执行结果：{result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"工具{tool_call['name']}执行结果：{result}"
                })

        return self.llm.invoke(messages, **kwargs)
