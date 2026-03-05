import os
import requests
import json
import dotenv

dotenv.load_dotenv()

def getWeather(city: str):
    params = {
        "key" : os.getenv("Weather_API_KEY"),
        "city" : city,
    }
    response = requests.get(os.getenv("Weather_URL"), params = params)
    data = response.json()
    return data

print(getWeather("北京"))