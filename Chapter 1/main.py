import json
from DeepSeek import DeepSeek
import Tools

agent_system_prompt = """
你是一个出行规划智能体，你的任务是根据用户的需求，自主规划和执行动作来制定出行计划。

你可以使用以下工具：
1. "getWeather": 获取指定城市的近期天气数据。参数：{"city": "<城市名称>"}
2. "summaryWeather": 总结获取到的天气情况。参数：{"weather": <获取到的天气数据列表，传入 getWeather 的完整返回结果>} 
3. "searchPlan": 根据目的地城市和天气总结文本，搜索推荐的游玩景点和计划。参数：{"city": "<城市名称>", "weather": "<天气总结文本>"}

【执行流程】
你需要通过连续的思考，决定下一步要采取的行动。请严格按照以下 JSON 的格式输出你的决定：
{
    "thought": "<你对当前情况的思考过程，分析需要使用哪个工具>",
    "action": "<要调用的工具名称，例如 getWeather / summaryWeather / searchPlan / FinalAnswer>",
    "action_input": {<调用该工具所需的参数字典，如果是 FinalAnswer，则应为包含最终计划文本的单字符串类型："最终给用户的回复文本">}
}

【注意】
- 请确保你的输出是一个合法的 JSON 格式字符串，纯 JSON 数据即可，请不要输出 Markdown 的 ```json 标记。
- 每次对话只能调用一个工具，等待系统返回工具执行结果（Observation）后，再继续思考和决定下一步。
- 在收集到所有需要的信息并且制定好包含游玩景点和天气注意事项的出行计划后，使用 "FinalAnswer" 动作向用户输出一份完整的、综合考虑了天气和景点的出行计划。
"""

def parse_agent_response(response: str):
    response = response.strip()
    # 尽可能清理 markdown 标记
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    return json.loads(response)

def run_agent(user_query: str):
    # 初始化大语言模型
    ds = DeepSeek(prompt=agent_system_prompt)
    
    print(f"========== 开始处理用户需求 ==========")
    print(f"User: {user_query}")
    print(f"===================================\n")
    
    # 将初始用户输入传入模型开始循环
    current_prompt = f"用户的需求是: {user_query}。请思考并开始使用工具。"
    
    max_steps = 10
    step = 0
    
    while step < max_steps:
        step += 1
        print(f"\n--- 步骤 {step} ---")
        
        response = ds.chat(current_prompt)
        
        try:
            command = parse_agent_response(response)
        except json.JSONDecodeError as e:
            print(f"[Agent Error] 输出格式错误，并非有效 JSON: \n{response}\n")
            # 尝试提示模型纠正自身格式
            current_prompt = "你的输出不是合法的纯 JSON 格式。请修正格式，重新输出仅包含符合预期的 JSON 字符串。"
            continue
            
        thought = command.get("thought", "")
        action = command.get("action", "")
        action_input = command.get("action_input", {})
        
        print(f"[Agent 思考]: {thought}")
        print(f"[Agent 行动]: 工具 {action} -> 参数 {str(action_input)[:200]}{'...' if len(str(action_input)) > 200 else ''}")
        
        # 判断是否得出最终结论
        if action == "FinalAnswer":
            print("\n========== 最终出行计划 ==========")
            print(action_input)
            break
            
        # 根据动作分发执行对应的工具
        result = None
        if action == "getWeather":
            city = action_input.get("city", "")
            if not city:
                result = "错误: 缺少 city 参数"
            else:
                result = Tools.getWeather(city)
                
        elif action == "summaryWeather":
            weather_data = action_input.get("weather", [])
            if not weather_data:
                result = "错误: 缺少 weather 参数"
            else:
                result = Tools.summaryWeather(weather_data)
                
        elif action == "searchPlan":
            city = action_input.get("city", "")
            weather_summary = action_input.get("weather", "")
            if not city or not weather_summary:
                result = "错误: searchPlan 缺少 city 或 weather 参数"
            else:
                result = Tools.searchPlan(city, weather_summary)
                
        else:
            result = f"错误: 发现未定义的工具调用 {action}"
            
        # 限制由于大量返回造成的刷屏
        result_str = str(result)
        print(f"[工具返回]: {result_str[:300]}{'...' if len(result_str) > 300 else ''}") 
        
        # 将工具执行结果作为 Observation 反馈给大模型用于下一步思考
        current_prompt = f"Observation (执行结果): {json.dumps(result, ensure_ascii=False)}"
    
    if step >= max_steps:
        print("\n[Agent 警告] 达到大语言模型的最大思考步数限制，未能及时完成任务。")

if __name__ == "__main__":
    # 执行测试用例
    user_query = "我想去深圳旅游，请帮我规划一下行程，另外注意看下近期的天气情况"
    run_agent(user_query)
