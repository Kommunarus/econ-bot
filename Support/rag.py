from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import re

def normaliz_cleanir_text(docs):
    for doc in docs:
        txt = doc.page_content
        text = txt.lower()
        text = re.sub(r"[\u00A0]+", " ", text)   # свертка пробелов
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"[\n ]{2,}", "\n\n", text)
        doc.page_content = text.strip()


def prepare_vdb():

    loader1 = PyPDFLoader("./data/Vinogradova.pdf")
    doc1 = loader1.load()
    normaliz_cleanir_text(doc1)


    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=['\n\n', '\n', '.'])

    chunks = []

    for doc in doc1:
        metadata = doc.metadata
        for chunk_text in splitter.split_text(doc.page_content):
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    print('Всего чанков: ', len(chunks))


    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local('./faiss/')

def process_docs(docs):
    return '\n\n'.join(['{} [{}, {}]'.format(doc.page_content, doc.metadata["source"], doc.metadata["page"]) for doc in docs])

def retr(query: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = FAISS.load_local('./faiss/', embeddings, allow_dangerous_deserialization=True)

    out = vector_store.similarity_search(query, k=10)

    return process_docs(out)


if __name__ == '__main__':
    prepare_vdb()

    pass