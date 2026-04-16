import os
import requests
import json
import dotenv
from DeepSeek import DeepSeek
from tavily import TavilyClient

dotenv.load_dotenv()

def getWeather(city: str):
    params = {
        "key" : os.getenv("Weather_API_KEY"),
        "city" : city,
        "extensions": "all",
    }
    response = requests.get(os.getenv("Weather_URL"), params = params)
    data = response.json()

    casts = data["forecasts"][0]["casts"]
    weather = [{
        "date": cast["date"],
        "dayweather": cast["dayweather"],
        "nightweather": cast["nightweather"],
        "daytemp": cast["daytemp"],
        "nighttemp": cast["nighttemp"],
    } for cast in casts]

    return weather


def summaryWeather(weather: list):
    sysPrompt = """
        你是一个天气总结助手，请根据为你提供的天气信息，总结未来几天的天气情况。
    """
    ds = DeepSeek(sysPrompt)
    res = ds.chat(", ".join(json.dumps(item) for item in weather))

    return res


def searchPlan(city: str, weather: str):
    tavily = TavilyClient(api_key = os.getenv("Tavily_API_KEY"))
    query = f"根据未来几天的天气情况：{weather}，搜索几个{city}的推荐游玩景点。"

    response = tavily.search(query = query, include_answer = "advanced")

    if response.get("answer"):
        return response["answer"]
    
    results = []
    for result in response.get("results", []):
        results.append(f"- {result['title']}: {result['content']}")
    
    if not results:
        return "抱歉，没有找到相关的旅游景点推荐。"

    return "根据搜索，为您找到以下信息:\n" + "\n".join(results)
