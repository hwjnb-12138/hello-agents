import re
from LLMClient import LLMClient
from ToolExecutor import ToolExecutor, search

# ReAct Prompt 模板：使用标准 str.format() 占位符，在实例化时统一注入工具列表
# 注意：prompt 中的花括号字面量需要用 {{ }} 转义
REACT_PROMPT_TEMPLATE = """你是一位智能助手，你的任务是根据用户需求，自主规划并执行动作来帮助用户。

你可以调用的工具有：
{tools}

你需要通过连续的思考，决定下一步要采取的行动，你必须严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
  - `{{tool_name}}[{{tool_input}}]`：调用一个可用工具，例如 Search[英伟达最新GPU型号]
  - `Finish[最终答案]`：当你已收集到足够信息，能够回答用户问题时使用此格式输出最终答案。

注意：每次回复必须且只能包含一个 Thought 和一个 Action。
"""

# 用户每轮输入的模板
USER_PROMPT_TEMPLATE = """现在请开始解决以下问题：
Question: {question}

历史记录（之前的思考和观察）：
{history}
"""


class ReActAgent:
    def __init__(self, llm: str, tool_executor: ToolExecutor, max_iterations: int = 5):
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        # 在实例化时将工具列表注入 system prompt
        system_prompt = REACT_PROMPT_TEMPLATE.format(
            tools=self.tool_executor.getAvaliableTools()
        )
        self.llm = LLMClient(model=llm, prompt=system_prompt)
        self._history_lines: list[str] = []

    def run(self, query: str) -> str | None:
        """
        运行 ReAct 循环，直到得出最终答案或达到最大迭代次数。

        Args:
            query: 用户的问题。

        Returns:
            最终答案字符串，或在超出迭代次数时返回 None。
        """
        print(f"\n{'='*20}")
        print(f"🤔 用户问题: {query}")
        print(f"{'='*20}")

        for current_step in range(1, self.max_iterations + 1):
            print(f"\n--- 步骤 {current_step} ---")

            # 构建本轮用户消息：包含原始问题和到目前为止的所有历史
            history_text = "\n".join(self._history_lines) if self._history_lines else "（暂无历史记录，这是第一步）"
            prompt = USER_PROMPT_TEMPLATE.format(
                question=query,
                history=history_text,
            )

            # 调用 LLM 进行思考
            response = self.llm.think(prompt)

            if not response:
                print("⚠️  LLM 返回了空响应，终止流程。")
                break

            # 解析 LLM 输出中的 Thought 和 Action
            thought, action = self._parse_output(response)

            if thought:
                print(f"\n💭 Thought: {thought}")
                self._history_lines.append(f"Thought: {thought}")

            if not action:
                print("⚠️  未找到有效的 Action，跳过本步骤。")
                self._history_lines.append("Observation: 未找到有效的 Action，请重新规划。")
                continue

            print(f"⚡ Action: {action}")

            # 判断是否为终止动作
            finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
            if finish_match:
                final_answer = finish_match.group(1).strip()
                print(f"\n✅ 最终答案: {final_answer}")
                return final_answer

            # 否则解析为工具调用
            tool_name, tool_input = self._parse_action(action)
            if tool_name and tool_input is not None:
                self._history_lines.append(f"Action: {tool_name}[{tool_input}]")
                tool_function = self.tool_executor.getTool(tool_name)
                if tool_function:
                    observation = tool_function(tool_input)
                else:
                    observation = f"错误：未找到名为 '{tool_name}' 的工具，请检查工具名称是否正确。"
                print(f"👁️  Observation: {observation}")
                self._history_lines.append(f"Observation: {observation}")
            else:
                error_msg = f"无法解析 Action 格式：'{action}'，请使用 tool_name[tool_input] 或 Finish[answer] 格式。"
                print(f"⚠️  {error_msg}")
                self._history_lines.append(f"Observation: {error_msg}")

        print("\n⛔ 已达到最大步数限制，流程终止。")
        return None

    def _parse_output(self, text: str) -> tuple[str | None, str | None]:
        """
        解析 LLM 的输出，提取 Thought 和 Action。
        使用 re.DOTALL 支持多行内容捕获，并取首个 Action 行。
        """
        # Thought 捕获到 Action: 之前的所有内容（支持多行 Thought）
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", text, re.DOTALL)
        # Action
        action_match = re.search(r"Action:\s*(.*?)(?=\nObservation:|\Z)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> tuple[str | None, str | None]:
        """
        解析 Action 字符串，提取工具名称和输入参数。
        支持工具输入包含嵌套括号或换行符。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None


if __name__ == '__main__':
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm = "deepseek-chat", tool_executor = tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)