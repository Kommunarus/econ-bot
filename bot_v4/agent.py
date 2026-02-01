from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import sys
import logging

from bot_v4.tools import make_tools
from bot_v4.caching import HookCacheMiddleware
from bot_v4.metrics import metrics_model_wrapper, metrics_tool_wrapper


def create_my_agent():
    load_dotenv('.env.example')
    MODEL = os.getenv('MODEL')
    API_KEY = os.getenv('API_KEY')
    API_BASE = os.getenv('API_BASE')

    model = ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.1,
        max_retries=3,
    )

    system_prompt = f'''Ты интеллектуальный ассистент ИТ компании.
    
    **Доступные инструменты:**
    
    1. **rag** — используй для вопросов о нашей компании, продуктах, услугах, ценах
       Примеры: "Что вы продаёте?", "Какие у вас гарантии?", "Расскажи о компании"
    
    2. **web_search** — используй для поиска актуальной общей информации, новостей, фактов
       Примеры: "Какая погода в Москве?", "Последние новости об AI", "Курс биткоина", "Последняя версия программы"
    
    3. **order_tracker** — используй ТОЛЬКО если пользователь указал номер заказа Order-XXXXXX
       Примеры: "Где мой заказ Ord-No:1234?", "Статус заказа Order-567890"
    
    4. **calculator** — для математических вычислений
       Примеры: "Посчитай 15% от 3000", "Сколько будет 45*12+78?"
    
    5. **currency** — для конвертации валют
       Примеры: "Переведи 100 USD в рубли", "Сколько стоит 50 EUR в долларах?"
    
    **Правила выбора инструмента:**
    - Если вопрос про нашу компанию/продукты → используй rag ПЕРВЫМ
    - Если вопрос общий или про актуальные события → используй web_search
    - Если видишь номер заказа → используй order_tracker БЕЗ других инструментов
    - НЕ используй web_search для вопросов о нашей компании
    - Можешь использовать несколько инструментов последовательно, если нужно
    
    Отвечай кратко, по делу. Всегда цитируй источники при использовании web_search.'''


    agent = create_agent(
        model=model,
        tools=make_tools(),
        system_prompt=system_prompt,
        middleware=[HookCacheMiddleware(), metrics_model_wrapper, metrics_tool_wrapper,
                    ToolRetryMiddleware(
                        max_retries=3,
                        backoff_factor=2.0,
                        initial_delay=1.0,
                        jitter=True,
                        retry_on=(ConnectionError, TimeoutError),
                        on_failure="return_message",
                        max_delay=60,
                    )
                    ]
    )

    return agent



if __name__ == '__main__':
    logging.basicConfig(
        filename="./logs/chat_session.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8"
    )
    logging.info("=== New session ===")
    n_total = 100
    n_current = 0
    agent = create_my_agent()

    while True:
        print("Вы: ", end="", flush=True)
        try:
            user_bytes = sys.stdin.buffer.readline()
            if not user_bytes:
                break
            user_text = user_bytes.decode('utf-8', errors='replace').strip()
        except KeyboardInterrupt:
            print("\nБот: До свидания!")
            break
        # user_text = 'Ответь на два вопроса: 1. Order 4587. 2. Найди в интернете информацию о компании 1С.'

        if not user_text:
            continue
        msg = user_text.lower()
        if msg in ("выход", "стоп", "конец"):
            print("Бот: До свидания!")
            break

        # result = agent.invoke({"messages": [{"role": "user", "content": msg}, ]})
        # print(result["messages"][-1].content)

        try:
            logging.info(f"Вы: {msg}")
            result = agent.invoke({"messages": [{"role": "user", "content": msg},]},
                                  {"recursion_limit": 11})
            print(result["messages"][-1].content)

            logging.info(f"Bot: {result["messages"][-1].content}")
        except Exception as e:
            print(f"Бот: [ошибка] {e}")
            logging.error(f"Error: {e}")

        n_current += 1
        if n_current >= n_total:
            break