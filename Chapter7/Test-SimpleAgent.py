from LLM import LLM
from SimpleAgent import SimpleAgent
from Tool import ToolRegistry
from Calculator import my_calculate

llm = LLM()
tool_registry = ToolRegistry()
tool_registry.register_function(
    name="calculator",
    description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
    function=my_calculate
)

dial_agent = SimpleAgent("对话智能体", llm, "你是一个智能助手，请友好回复用户的问题")
response1 = dial_agent.run("今天上海天气如何？")
print(f"简单对话测试：{response1}\n")

tool_agent = SimpleAgent(
    name="工具智能体", 
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户",
    enable_tool_calling=True,
    tool_registry=tool_registry
)
response2 = tool_agent.run("请调用工具帮我计算5 + 2 * 3")
print(f"工具智能体测试：{response2}\n")

print("流式对话测试：")
response3 = dial_agent.stream_run("hello!")
for chunk in response3:
    pass  # 生成器已经在内部打印了内容
print()

print("列出所有工具：", dial_agent.list_tools())
dial_agent.add_tool(name="calculator", description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数", func = my_calculate)
print("列出所有工具：", dial_agent.list_tools())