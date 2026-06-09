from MemoryTool import MemoryTool

try:
    from ..Chapter7.Tool import ToolRegistry
    from ..Chapter7.SimpleAgent import SimpleAgent
    from ..Chapter7.LLM import LLM
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter7'))
    from Chapter7.Tool import ToolRegistry
    from Chapter7.ReActAgent import ReActAgent
    from Chapter7.LLM import LLM

def main():
    print("=" * 60)
    print("测试 MemoryTool 记忆工具功能")
    print("=" * 60)

    print("\n1. 初始化组件...")
    llm = LLM()
    tool_registry = ToolRegistry()
    memory_tool = MemoryTool(memory_types=["working"])
    tool_registry.register_tool(memory_tool)
    print("LLM 初始化完成")
    print(f"工具注册完成: {tool_registry.list_tools()}")

    print("\n2. 创建记忆型智能体...")
    agent = ReActAgent(
        name="记忆助手",
        llm=llm,
        tool_registry=tool_registry,
        max_iterations=3
    )
    print("智能体创建完成")

    print("\n" + "-" * 60)
    print("测试 1: 记忆存储")
    print("-" * 60)
    response1 = agent.run("记住：我爱吃西瓜")
    print(f"\n📝 回答:\n{response1}")

    print("\n" + "-" * 60)
    print("测试 2: 记忆检索")
    print("-" * 60)
    response2 = agent.run("我爱吃西瓜吗？")
    print(f"\n📝 回答:\n{response2}")

    print("\n" + "-" * 60)
    print("测试 3: 更新记忆")
    print("-" * 60)
    response3 = agent.run("我不爱吃西瓜了，我现在想吃苹果")
    print(f"\n📝 回答:\n{response3}")

    print("\n" + "-" * 60)
    print("测试 4: 查看记忆统计")
    print("-" * 60)
    response4 = agent.run("现在我有多少条记忆了？给我看看统计信息")
    print(f"\n📝 回答:\n{response4}")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()