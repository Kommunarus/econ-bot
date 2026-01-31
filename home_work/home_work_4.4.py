import os
from langchain.agents.middleware.types import AgentMiddleware, hook_config, StateT
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Dict, Any
import hashlib
import time
import json
import pickle

from dotenv import load_dotenv


load_dotenv('my.env')
MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')

model = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_BASE, temperature=0.3)


class CachedRetriever(BaseRetriever):
    vectorstore: FAISS
    cache: Dict = {}
    ttl_seconds: int = 300
    k: int = 3
    path_cache:str = 'cache_retriever.pkl'

    def _save_cache(self):
        with open(self.path_cache, 'wb') as f:
            # json.dump(self.cache, f, indent=4, ensure_ascii=False)
            pickle.dump(self.cache, f)
            print('Save cache retriever')

    def _load_cache(self):
        if os.path.exists(self.path_cache):
            with open(self.path_cache, 'rb') as f:
                # self.cache = json.load(f)
                self.cache = pickle.load(f)
                print('Load cache retriever')
        else:
            self.cache = {}

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _get_relevant_documents(self, query) -> list[Document]:
        key = self._hash_query(query)
        now = time.time()

        self._load_cache()
        if key in self.cache:
            timestamp, results = self.cache[key]
            age = now - timestamp
            if age > self.ttl_seconds:
                del self.cache[key]
            else:
                return results

        results = self.vectorstore.similarity_search(query, self.k)
        self.cache[key] = (now, results)
        self._save_cache()
        return results

    def cache_invalidate(self, query):
        key = self._hash_query(query)
        if self.cache.get(key) is not None:
            del self.cache[key]


class HookCacheMiddleware(AgentMiddleware):
    def __init__(self, path_cache='cache_llm.json'):
        self.path_cache = path_cache
        self.ttl_seconds = 300

        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.path_cache):
            with open(os.path.join(self.path_cache), 'r') as f:
                self.cache = json.load(f)
                print('Load cache llm')
        else:
            self.cache = {}

    def _save_cache(self):
        with open(self.path_cache, 'w') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)
            print('Save cache llm')

    def _hash_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode('utf-8')).hexdigest()

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        input_text = [m for m in state.get("messages") if isinstance(m, HumanMessage)][-1].content
        tool_msg = [m for m in state.get("messages") if isinstance(m, ToolMessage)]
        self.key = self._hash_key(input_text)
        now = time.time()

        if self.key in self.cache:
            timestamp, cached_answer = self.cache[self.key]
            age = now - timestamp
            if age > self.ttl_seconds:
                del self.cache[self.key]
            else:
                if len(tool_msg) == 0:
                    return {"jump_to": "end", "messages": [AIMessage(content=cached_answer)]}
        return None

    def after_model(self, state, runtime):
        now = time.time()
        answer_text = [msg for msg in state.get("messages") if isinstance(msg, AIMessage)][-1].content
        self.cache[self.key] = (now, answer_text)
        self._save_cache()
        return None

    def cache_invalidate(self, query):
        key = self._hash_key(query)
        if self.cache.get(key) is not None:
            del self.cache[key]


docs = [
    Document(page_content='Саша любит Наташу'),
    Document(page_content='Наташа любит Сашу'),
    Document(page_content='Оля любит снег'),
    Document(page_content='Коля любит игры'),
    Document(page_content='Варя любит Диму'),
    Document(page_content='Дима любит Дашу'),
    Document(page_content='Даша любит Варю'),
]

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = CachedRetriever(vectorstore=vectorstore)

@tool
def tool_retriever(text: str) -> str:
    """Поиск информации в базе знаний о личностных предпочтениях"""
    out = retriever.invoke(text)
    return '\n'.join([doc.page_content for doc in out])


agent = create_agent(
    model=model,
    tools=[tool_retriever],
    middleware=[HookCacheMiddleware()]
)

query = 'Кого Оля любит?'
start = time.time()
response = agent.invoke({"messages": [{"role": "user", "content": query},]})
print(response["messages"][-1].content)
elapsed1 = time.time() - start
print(f"Время выполнения: {elapsed1:.6f} секунд")

start = time.time()
response = agent.invoke({"messages": [{"role": "user", "content": query},]})
print(response["messages"][-1].content)
elapsed2 = time.time() - start
print(f"Время выполнения: {elapsed2:.6f} секунд")

# Первый запуск
# Save cache llm
# Save cache retriever
# Save cache llm
# **Ответ:** Оля любит **снег**.
# Время выполнения: 0.884721 секунд
# **Ответ:** Оля любит **снег**.
# Время выполнения: 0.002298 секунд


# Второй запуск
# Load cache llm
# **Ответ:** Оля любит **снег**.
# Время выполнения: 0.003912 секунд
# **Ответ:** Оля любит **снег**.
# Время выполнения: 0.001593 секунд

# проблема в том, что агент ллм запускается два раза.
# первый раз он возвращает ai_msg пустое и запускает тулз. и если в это время сохранить кеш,
# то когда агент запустит ллм второй раз, то ответ сразу считается с кеша , а он там пустой.
# поэтому пришлось сделать еще одно условия что, если сообщение tool_msg не пустое, то не читать кеш.
# в первом вызове ллм tool_msg пустое, а во втором разе - уже есть три записи из бд.
# и значит кеш записывается 2 раза, но используется только после второго запуска скрипта.