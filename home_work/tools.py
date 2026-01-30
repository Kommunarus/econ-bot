from langchain.tools import tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

import ast
import math

import requests

def geocode(city: str):
    '''Получает координаты по названию города'''
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": city, "format": "json", "limit": 1},
        headers={"User-Agent": "weather-agent/1.0"}
    )
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])

@tool("Calculator")
def calculator(sample: str) -> str:
    """Инструмент выполняет арифметические выражения.
    Разрешены только базовые операции и функции из math
    Пример: sample='(6+9/2)**2'"""
    tree = ast.parse(sample, mode='eval')

    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith('_')
    }
    allowed_names.update({
        'abs': abs,
        'max': max,
        'min': min,
        'pow': pow,
        'round': round,
    })

    code = compile(tree, filename='', mode='eval')
    try:
        answer = eval(code, {"__builtins__": {}}, allowed_names)
    except Exception as f:
        print('error: ', f)
        answer = f
    return str(answer)

@tool("weather")
def weather(city: str) -> str:
    """Возвращает текущую температуру в указанном городе."""
    coords = geocode(city)
    if not coords:
        return f"Город «{city}» не найден."
    lat, lon = coords
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=5
    )
    data = resp.json()
    cw = data.get("current_weather")
    if not cw:
        return "Не удалось получить текущую погоду."
    return f"В городе {city} сейчас {cw['temperature']}°C."

@tool("read_file")
def read_file(path: str) -> str:
    """Возвращает содержимое текстового файла.
    На входе - имя файла. Файл ищется в папке docs.
    Пример: path='test.txt'"""
    with open(os.path.join('./docs', path), 'r') as f:
        try:
            text = f.read()
        except Exception as e:
            text = e

    return text

@tool("fx_rate")
def fx_rate(from_currency: str) -> str:
    """
    Получает актуальный курс валюты в рублях за одну единицу базовой валюты.

    Пример: from_currency='USD'
    """
    from_currency = from_currency.upper()
    to_currency = 'RUB'

    try:
        resp = requests.get("https://api.fxratesapi.com/latest", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates")
        if not rates:
            return f"Не удалось получить курсы валют: {data}"

        if from_currency not in rates:
            return f"Неизвестная валюта-источник: {from_currency}"
        if to_currency not in rates:
            return f"Неизвестная валюта-назначение: {to_currency}"

        # конвертация через базовую валюту API (USD)
        amount_in_usd = 1 / rates[from_currency]  # переводим в USD
        converted_amount = round(amount_in_usd * rates[to_currency], 2)

        return f"{1} {from_currency} ≈ {converted_amount} {to_currency}"

    except Exception as e:
        return f"Ошибка при запросе курса: {e}"


def make_tools():
    return [calculator, weather, read_file, fx_rate]


if __name__ == '__main__':
    load_dotenv('my.env')
    MODEL = os.getenv("MODEL")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")

    model = ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.05,
        max_retries=2
    )

    system_prompt = """
    Ты агент-помощник.
    Если пользователь просит посчитать что-либо — ты ВСЕГДА вызываешь инструмент Calculator.
    Всегда отвечай пользователю тем, что получил из инструмента калькулятор - не считай сам.
    
    Если тебя спрашивают про погоду в городе, ВСЕГДА используй инструмент weather.
    
    Если тебя просят прочитать файл с диска, ВСЕГДА используй инструмент read_file 

    Если тебя спрашивают про курс валюты, ВСЕГДА используй инструмент fx_rate 
    """

    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[calculator, weather, read_file, fx_rate],
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                jitter=True,
                retry_on=(ConnectionError, TimeoutError),
                on_failure="return_message",
                max_delay=60,
            ),
        ],
    )

    inp = input('Вы: ')
    response = agent.invoke({"messages": [{
        "role": "user",
        "content": inp}],},
{"recursion_limit": 11} )

    print(response['messages'][-1].content)  # '5'