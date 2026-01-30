import os
from typing import Dict, Set
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS
import hashlib

import time


search_cache: Dict[str, str] = {}
user_search_cache: Dict[str, str] = {}
set_uid: Set[str] = set()

@tool("web_search", return_direct=False)
def web_search(query: str, user_query:str, k: int = 5) -> str:
    """Выполняет веб-поиск. Возвращает top-k результатов
    query - запрос агента, возможно измененный и дополненный относительно того, что спрашивал пользователь.
    user_query - оригинальный вопрос от пользователя, без изменений.
    k - сколько ссылок нужно возвращать по запросу query"""

    print('Вызов инструмента')
    print(query)
    print(user_query)
    uid = hashlib.md5(query.encode("utf-8")).hexdigest()

    if uid in set_uid:
        print('use cache')
        return search_cache[uid]

    ddgs = DDGS()

    results = []

    for i, r in enumerate(ddgs.text(query, max_results=k, timelimit="y")):
        title = r.get("title")
        snippet = r.get("body") or r.get("Text") or ""
        url = r.get("href")
        results.append(f"{i + 1}. {title}: {snippet} [source: {url}]")


    output = "\n".join(results)
    search_cache[uid] = output
    user_search_cache[user_query] = output
    set_uid.add(uid)
    return output



if __name__ == '__main__':
    load_dotenv('my.env')
    MODEL = os.getenv("MODEL")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")

    model = ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.05
    )

    model_checker = ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.05
    )

    prompt_checker = ChatPromptTemplate.from_template("""Ты эксперт по оценке качества ответов.
                                            Вопрос: {query}
                                            Контекст: {search_results}
                                            Полученный ответ: {prediction}
                                            Оцени, насколько полученный ответ соответствует вопросу по фактам, указанным в контексте.
                                            Ответь ТОЛЬКО одним словом: CORRECT или INCORRECT. 
                                            Даже ОДНА ошибка в ответе по фактам, должна приводить к оценке INCORRECT"""
                                        )

    system_prompt = """
    Ты агент-помощник. У тебя есть следующие инструменты:
    web_search: поиск в интернете
    Если пользователь задаёт фактологический вопрос — сначала попробуй web_search.
    Отдавай приоритет официальным источникам информации.
    Всегда указывай источники. В тексте рядом с цитатой в квадратных скобках ставь индекс цитирования, а в конце ответа перечисли список источников в соответствии с этими индексами.
    Индексацию цитирования начинай с 1 и в порядке появления цитаты в тексте ответа, а не по индексам из поисковика.
    например:
    текст ответа с фактом один [1] и продолжение текста с фактом два [2]
    источники:
    [1] ссылка на источник факта один
    [2] ссылка на источник факта два
    """

    agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt=system_prompt,
    )
    n = 0
    q = "найди число жителей земли на данный момент"
    while n<1:
        s = time.time()
        response = agent.invoke({"messages": [{"role": "user",
                                               "content": q}]})

        # print(response["messages"][-1].content)
        prediction = response["messages"][-1].content
        print(prediction)
        e = time.time()
        print('Время выполнения, сек: ', round(e-s, 1))
        n += 1

        search_results = user_search_cache.get(q, 'Контекст не был предоставлен. Ответ возможно некорректный')
        print(search_results)
        prompt = prompt_checker.format_prompt(query=q, prediction=prediction, search_results=search_results)
        response = model_checker.invoke(prompt)
        verdict = response.content.strip().upper()
        score = 1.0 if verdict == "CORRECT" else 0.0
        print('score inspector', score)

        # '''
        # сделан бот, который может получать контекст с интернета
        # дополнительно добавлено:
        # кеш хеш,
        # проверка ответа второй сеткой
        # '''

        # Пример вывода:
        #
        # Вызов инструмента
        # current world population
        # найди число жителей земли на данный момент
        # Число жителей Земли на данный момент составляет примерно **8 280 000 000 человек** (≈ 8,28 млрд) [1].
        #
        # ---
        #
        # ### Источники
        # [1] https://georank.org/population
        #
        # (Данные основаны на последних оценках ООН и актуализируются в реальном времени.)
        # Время выполнения, сек:  3.5
        # 1. World Population (2026): The current state of the world population with real-time data and analysis on population growth, demographics, and more. [source: https://populationtoday.com/]
        # 2. World population live, January 2026 - Nations Geo: The current world population of 2026 There are 8,271,411,651 people on Earth today (8.2714 billion) on Wednesday, January 28, 2026, with a growth rate of 0.82% per year, 363,023 births daily, 174,349 deaths daily, and 186,832 population increases daily. The world population is growing at a rate of almost 130 people per minute. [source: https://nationsgeo.com/population/]
        # 3. U.S. and World Population Clock - Census.gov: Shows estimates of current USA Population overall and people by US state/county and of World Population overall, by country and most populated countries. [source: https://www.census.gov/popclock/]
        # 4. Current World Population Clock: 197 country counters [2026]: The current world population is 8,280,172,104 people, as of 2026 based on the UN estimation and the latest data. [source: https://georank.org/population]
        # 5. World Population 2026 - Live Counter, Demographics, Religion, Literacy ...: Live world population counter with comprehensive demographics: age structure, literacy, religion, urbanization, continent and country breakdowns, and projections to 2100. Sources: UN WPP, World Bank, Pew Research. [source: https://countrymeters.info/en/World]
        # score inspector 1.0
