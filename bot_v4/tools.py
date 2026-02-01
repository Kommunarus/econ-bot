import datetime
from typing import Set, Dict

import ast

import hashlib
import requests
import random
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from ddgs import DDGS

from bot_v4.caching import CachedRetriever


def read_docs():
    docs = []
    loader1 = TextLoader('./knowledge_base/company_info.txt')
    loader2 = TextLoader('./knowledge_base/products.txt')
    docs.append(loader1.load()[0])
    docs.append(loader2.load()[0])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40, )
    splitted_docs = text_splitter.split_documents(docs)
    return splitted_docs


@tool("calculator")
def calculator(sample: str) -> str:
    """Инструмент выполняет арифметические выражения.
    Разрешены только базовые операции и функции из math
    Пример: sample='(6+9/2)**2'"""
    import math
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
        print('error calc: ', f)
        answer = f
    return str(answer)


search_cache: Dict[str, str] = {}
user_search_cache: Dict[str, str] = {}
set_uid: Set[str] = set()

@tool("web_search", return_direct=False)
def web_search(query: str, user_query:str, k: int = 5) -> str:
    """Выполняет веб-поиск. Возвращает top-k результатов
    query - запрос агента, возможно измененный и дополненный относительно того, что спрашивал пользователь.
    user_query - оригинальный вопрос от пользователя, без изменений.
    k - сколько ссылок нужно возвращать по запросу query"""

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

@tool("currency")
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из from_currency в to_currency по актуальному курсу.

    Пример: amount=10, from_currency='USD', to_currency='RUB'
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

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
        amount_in_usd = amount / rates[from_currency]  # переводим в USD
        converted_amount = round(amount_in_usd * rates[to_currency], 2)

        return f"{amount} {from_currency} ≈ {converted_amount} {to_currency} (курс {to_currency}: {rates[to_currency]}, курс {from_currency}: {rates[from_currency]})"

    except Exception as e:
        return f"Ошибка при запросе курса: {e}"

embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
docs = read_docs()
vectorstore = FAISS.from_documents(docs, embeddings)
cache_retriever = CachedRetriever(vectorstore=vectorstore)

@tool('rag')
def retriever(text: str) -> str:
    """Поиск по документам компании"""
    out = cache_retriever.invoke(text)
    return '\n'.join([doc.page_content for doc in out])

@tool('order_tracker')
def order_tracker(text: str) -> str:
    '''Отслеживание по номеру заказа. Возвращает статус, дату доставки и текущее местоположение'''
    status = random.choice(['Создан', 'Ожидает оплаты ', 'В сборке', 'Передаётся в доставку',
                            'В пути', 'У курьера', 'Ожидает получения', 'Доставлено',
                            'Получен', 'Отменено'])

    # Генерируем случайную дату между двумя заданными датами
    start_date = datetime.date(2026, 1, 1)
    delivery_date = (start_date + datetime.timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
    current_location = random.choice(["Москва", "Питер", "Новосибирск"])
    return f'{status} {current_location} {delivery_date}'


def make_tools():
    return [calculator, web_search, currency_converter, retriever, order_tracker]