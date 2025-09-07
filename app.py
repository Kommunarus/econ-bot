from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_community.callbacks import get_openai_callback

import logging
import json

from dotenv import load_dotenv
import os
import datetime

if os.path.exists('./logs') == False:
    os.mkdir('./logs')

logging.basicConfig(
    filename=f"./logs/session_{datetime.datetime.now()}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
load_dotenv('my.env')



def main():
    # Настройка доступа: загрузка переменных окружения, установка ключа и адреса API

    # Создаём модель и память
    chat_model = ChatOpenAI(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("API_BASE"),
        model=os.getenv("MODEL"),
        temperature=0.1,
        request_timeout=15)

    memory = ConversationBufferMemory()

    memory.chat_memory.add_message(
        SystemMessage(content="Ты бот поддержки магазина «Shoply»")
    )
    memory.chat_memory.add_message(
        SystemMessage(content="Отвечай кратко и по делу. Если не уверен — так и скажи.")
    )

    # Цепочка диалога на основе модели и памяти
    conversation = ConversationChain(llm=chat_model, memory=memory)

    with open("./data/faq.json") as f:
        faq_data = json.load(f)
    with open("./data/orders.json") as f:
        orders = json.load(f)


    memory.chat_memory.add_message(
        SystemMessage(content=f"Тебе будут задавать вопросы. Отвечай из этого вопросника: {faq_data}")
    )
    memory.chat_memory.add_message(
        SystemMessage(content=f"Информация о существующих заказах: {orders}")
    )
    memory.chat_memory.add_message(
        SystemMessage(content=f"Если тебя спросят в формате '/order <id>', сообщи статус по заказу id или откажи, сказав что такого заказа нет.")
    )

    print("Привет! Я бот поддержки магазина «Shoply». Для выхода напишите «выход».")
    logging.info("=== New session ===")
    chat_loop(conversation)

def chat_loop(conversation: ConversationChain):
    while True:
        try:
            user_text = input("Вы: ")
        except (KeyboardInterrupt, EOFError):
            print("\nБот: Завершение работы.")
            break

        user_text = user_text.strip()
        if user_text == "":
            continue

        logging.info(f"User: {user_text}")

        commandas = user_text.lower()
        if commandas in ("выход", "стоп", "конец"):
            print("Бот: До свидания!")
            break
        if commandas in ("сброс",):
            conversation.memory.clear()
            print("Бот: Контекст диалога очищен.")
            continue



        try:
            with get_openai_callback() as cb:
                reply = conversation.predict(input=user_text)
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens
                total_tokens = cb.total_tokens

            pass
        except Exception as e:
            print(f"Бот: [Ошибка] {e}")
            logging.error(f"Error: {e}")
            continue
        reply = reply.strip()
        logging.info(f"Bot: {reply}")
        logging.info(f"Usage (prompt/completion/total ): {prompt_tokens}/{completion_tokens}/{total_tokens}")



        print(f"Бот: {reply}", "\n")


if __name__ == "__main__":
    main()