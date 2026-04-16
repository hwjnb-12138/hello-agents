from LLMClient import LLMClient
from typing import List, Dict, Any, Optional

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
```
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

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

class ReflectionAgent:
    def __init__(self, llm_client: LLMClient, max_iterations: int = 3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations
    
    def run(self, task: str):
        print(f"---------- 开始执行任务:{task} ----------")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task = task)
        response = self.llm_client.think(initial_prompt)
        self.memory.add_memory("execution", response)

        for i in range(self.max_iterations):
            print(f"---------- 第{i}次迭代 ----------")

            print("正在进行反思...")
            reflection_prompt = REFLECT_PROMPT_TEMPLATE.format(task = task, code = self.memory.get_last_execution())
            reflection_response = self.llm_client.think(reflection_prompt)
            self.memory.add_memory("reflection", reflection_response)

            if "无需改进" in reflection_response:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            print("正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task = task,
                last_code_attempt = self.memory.get_last_execution(),
                feedback = reflection_response
            )
            refine_response = self.llm_client.think(refine_prompt)
            self.memory.add_memory("execution", refine_response)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

if __name__ == '__main__':
    llm_client = LLMClient()
    agent = ReflectionAgent(llm_client, max_iterations=2)
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)