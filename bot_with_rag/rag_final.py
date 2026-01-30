import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.retriever import hybrid_retriver, check_confidence, format_docs
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
import logging

logging.basicConfig(
    filename="./logs/chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

Hybrid_retriever = hybrid_retriver()



logging.basicConfig(
    filename="./logs/chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

load_dotenv('.env.example')

MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

llm = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    temperature=0.1,
)

session_store = {}
def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Ты знаток творчества Омара Хайяма. "
    "Отвечай по теме творчества Омара Хайяма ТОЛЬКО на основе приведенного контекста. Плюс учитывай историю разговора. "
    "Если тебя просят написать рубайю - пиши ее полностью из контекста. Ничего не выдумывай. "
    "Если тебя спрашивают общие вопросы по творчеству Омара Хайяма - пиши кратко, но все только по контексту + учитывай "
    "историю сообщений."
    "Если контекста нет - ответь, что не знаешь ответа. Это означает что в базе знаний ничего не нашлось по "
    "запрашиваемой теме. "
    ""),
    MessagesPlaceholder(variable_name="history"),
    ("user", "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:")
])

check_step = RunnableLambda(check_confidence)

chain = (
    RunnablePassthrough.assign(
            context=itemgetter("question") | Hybrid_retriever | check_step | format_docs
        )
    | prompt
    | llm
)

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",    # ключ для текущего вопроса
    history_messages_key="history",   # ключ для истории в prompt
)


if __name__ == "__main__":
    session_id = "user_1"
    logging.info("=== New session ===")
    while True:
        # user_text = input("Вы: ").strip()
        print("Вы: ", end="", flush=True)
        try:
            user_bytes = sys.stdin.buffer.readline()
            if not user_bytes:
                break
            user_text = user_bytes.decode('utf-8', errors='replace').strip()
        except KeyboardInterrupt:
            print("\nБот: До свидания!")
            break

        if not user_text:
            continue
        msg = user_text.lower()
        if msg in ("выход", "стоп", "конец"):
            print("Бот: До свидания!")
            break

        try:
            result = chain_with_memory.invoke(
                {"question": user_text},
                config={"configurable": {"session_id": session_id}}
            )
            print(result.content)

            logging.info(f"Bot: {result.content}")
        except Exception as e:
            print(f"Бот: [ошибка] {e}")
            logging.error(f"Error: {e}")