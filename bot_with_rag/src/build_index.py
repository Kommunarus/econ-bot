from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import re
import pickle

def normaliz_cleanir_text(docs):
    for doc in docs:
        txt = doc.page_content
        text = txt.lower()
        text = re.sub(r"[\u00A0]+", " ", text)   # свертка пробелов
        text = re.sub(r"[\n ]{2,}", "\n\n", text)
        doc.page_content = text.strip()


# Оригиналы файлов взяты с сайта озон. Изменено название компании и убраны картинки
loader1 = TextLoader("./data/Рубаи.txt")
doc1 = loader1.load()
normaliz_cleanir_text(doc1)



splitters = [
                RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=['\n\n', '\n', '.']),
                # RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60),
                # RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=90),
]

chunks = []
for splitter in splitters:
    for doc in doc1:
        metadata = doc.metadata
        for chunk_text in splitter.split_text(doc.page_content):
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

print('Всего чанков: ', len(chunks))


embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.save_local('./faiss/')

# Сохранение документов для bm25
with open("./docdb/bm25_docs.pkl", "wb") as f:
    pickle.dump(chunks, f)
