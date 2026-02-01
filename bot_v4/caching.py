# class RAGCache:
#     """Кэширование эмбеддингов и результатов поиска"""
#     - Хранение эмбеддингов запросов
#     - Кэш результатов поиска (top-N документов)
#     - TTL для автоматической инвалидации
#     - Сохранение в файл и загрузка при инициализации
#     - Метод cache_invalidate(query)
#
# class LLMCache:
#     """Кэширование ответов модели"""
#     - Хэширование запросов (учёт системного промпта + сообщения)
#     - Сохранение AIMessage с метаданными
#     - TTL для каждой записи
#     - Middleware для перехвата запросов
#     - Персистентность (сохранение в JSON)
import os
import pickle
import json
import time
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import Set, Dict
import hashlib
from langchain_community.vectorstores import FAISS
from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class CachedRetriever(BaseRetriever):
    vectorstore: FAISS
    cache: Dict = {}
    ttl_seconds: int = 3600
    k: int = 5
    path_cache:str = './cache/cache_retriever.pkl'

    def _save_cache(self):
        with open(self.path_cache, 'wb') as f:
            pickle.dump(self.cache, f)
            # print('Save cache retriever')

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
    def __init__(self, path_cache='./cache/cache_llm.json'):
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
            # print('Save cache llm')

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