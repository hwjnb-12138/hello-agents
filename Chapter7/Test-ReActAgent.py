# -*- coding: utf-8 -*-
"""
测试 ReActAgent 智能体功能的脚本
"""

from LLM import LLM
from ReActAgent import ReActAgent
from Calculator import my_calculate


def test_react_agent():
    print("=" * 60)
    print("测试 ReActAgent 智能体功能")
    print("=" * 60)
    
    # 1. 初始化 LLM
    print("\n1. 初始化 LLM...")
    llm = LLM()
    print("LLM 初始化完成")
    
    # 2. 创建 ReActAgent
    print("\n2. 创建 ReActAgent...")
    agent = ReActAgent(
        name="ReAct智能体",
        llm=llm,
        max_iterations=5
    )
    print("ReActAgent 初始化完成")
    
    # 3. 添加测试工具
    print("\n3. 添加测试工具...")
    agent.add_tool(
        name="calculator",
        description="数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=my_calculate
    )
    print("工具添加完成")
    
    # 4. 测试基本对话
    print("\n" + "-" * 60)
    print("4. 测试基本对话")
    print("-" * 60)
    response1 = agent.run("你好，介绍一下你自己")
    print(f"智能体回答：{response1}")
    
    # 5. 测试工具调用 - 数学计算
    print("\n" + "-" * 60)
    print("5. 测试工具调用 - 数学计算")
    print("-" * 60)
    response2 = agent.run("请计算 2 + 3 * 4")
    print(f"智能体回答：{response2}")
    
    # 6. 测试多步骤推理
    print("\n" + "-" * 60)
    print("6. 测试多步骤推理")
    print("-" * 60)
    response3 = agent.run("请先计算 10 的平方根，然后将结果乘以 2")
    print(f"智能体回答：{response3}")
    
    # 7. 测试复杂问题
    print("\n" + "-" * 60)
    print("7. 测试复杂问题")
    print("-" * 60)
    response4 = agent.run("我有5个苹果，吃了2个，又买了3个，现在有几个苹果？")
    print(f"智能体回答：{response4}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_react_agent()