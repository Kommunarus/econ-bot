import pickle

from langchain_core.retrievers import BaseRetriever
import hashlib
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

class SimpleReranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=3):

        if not documents:
            return []

        # Подготавливаем пары (query, document)
        pairs = [[query, doc.page_content] for doc in documents]

        # Получаем скоры релевантности
        scores = self.model.predict(pairs)

        # Сортируем документы по скору
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Отбираем топ-N документов (score записываем в метаданные)
        result = []
        for doc, score in doc_score_pairs[:top_n]:
            doc.metadata = doc.metadata or {}
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)

        return result


class HybridRerankerRetriever(BaseRetriever):
    first_retriever: BaseRetriever
    second_retriever: BaseRetriever
    reranker: SimpleReranker
    k: int = 3

    def _dedup_docs(self, docs):
        '''объединение результатов с дедупликацией'''
        seen = set()
        result = []
        for d in docs:
            content = d.page_content
            uid = hashlib.md5(content.encode("utf-8")).hexdigest()
            if uid not in seen:
                seen.add(uid)
                result.append(d)
        return result

    def _get_relevant_documents(self, query):
        first_docs = self.first_retriever.invoke(query)
        second_docs = self.second_retriever.invoke(query)
        merged = self._dedup_docs(first_docs + second_docs)

        return self.reranker.rerank(query, merged, self.k)

def hybrid_retriver():

    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    vector_db = FAISS.load_local("./faiss/", embed_model,
                                              allow_dangerous_deserialization=True)

    with open("./docdb/bm25_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    # инициализация гибридного retriever+reranker
    retriver = HybridRerankerRetriever(
        first_retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        second_retriever=BM25Retriever.from_documents(docs, k=5),
        reranker=SimpleReranker("BAAI/bge-reranker-base"),
        k=3
        )

    return retriver

def format_docs(docs):
    """Форматирование документов в строку"""
    context_texts = [f"{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(context_texts)

def check_confidence(inputs, threshold=0.25):
    conf_docs = []
    for doc in inputs:
        score = doc.metadata.get('rerank_score', 0.0)
        if score >= threshold:
            conf_docs.append(doc)

    inputs = conf_docs
    return inputs


if __name__ == '__main__':
    results = hybrid_retriver().invoke("почему омар хайям популярен")

    print("=== Гибридный BM25 + Vector + Rerank ===")
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("rerank_score", 0)
        print(f"{i}. {score:.4f} | {doc.page_content}")

