from LLM import LLM
from SimpleAgent import SimpleAgent
from Tool import ToolRegistry
from Calculator import my_calculate

llm = LLM()
tool_registry = ToolRegistry()
tool_registry.register_function(
    name="my_calculator",
    description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
    function=my_calculate
)

dial_agent = SimpleAgent("对话智能体", llm, "你是一个智能助手，请友好回复用户的问题")
response1 = dial_agent.run("今天上海天气如何？")
print(f"简单对话测试：{response1}")
