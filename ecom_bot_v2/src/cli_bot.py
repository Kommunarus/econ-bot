from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
import logging
from pydantic import BaseModel, Field
import yaml

with open("../data/style_guide.yaml", "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)

logging.basicConfig(
    filename="chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'my.env'))

MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

chat_model = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    temperature=0.4,
    request_timeout=15,
)

class AnswerInfo(BaseModel):
    answer: str = Field(..., description=STYLE['format']['fields']['answer'])
    tone: str = Field(..., description=STYLE['format']['fields']['tone'])
    actions: list[str] = Field(..., description=STYLE['format']['fields']['actions'])

output_parser = PydanticOutputParser(pydantic_object=AnswerInfo)
format_instructions = output_parser.get_format_instructions()

# Создаём класс для CLI-бота
class CliBot():
    def __init__(self):
        # Создаём модель
        self.chat_model = chat_model

        # Создаём Хранилище истории
        self.store = {}

        # Читаем внешние файлы. В них все заказы и примеры ответов на вопросы.
        self.faq, self.orders, self.few_shots = self.read_files()


        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты консультант в магазине. Твоя задача отвечать на вопросы по заказам покупателей. "
                       "Вот примеры вопросов и ответов. Следуй этим примерам, когда отвечаешь на вопросы. \n"
                       "ПРИМЕРЫ: {faq}\n"
                       "Если пользователь пришлет сообщение следующего вида: **/order <id>**, дай статус"
                       "по заказу, используя эти данный из базы данных: {orders}. \n"
                       "Если заказа не окажется в базе данных, дай вежливый отказ и сообщи, что данных "
                       "о заказе пока нет. " + STYLE['fallback']['no_data']),
            ("system", f"Тон ответов: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
                       f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."),
            ("system", "Далее приведены примеры вопросов и примеры правильных ответов. \n{few_shots}\n"),
            ("system", "Отвечай в требуемом формате. \n{format_instructions}\n"),
             MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку (тут используется синтаксис LCEL*)
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,  # Цепочка с историей
            self.get_session_history,  # метод для получения истории
            input_messages_key="question",  # ключ для вопроса
            history_messages_key="history",  # ключ для истории
        )

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def read_files(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'data/faq.json'), 'r', encoding='utf-8') as f:
            faq = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data/orders.json'), 'r', encoding='utf-8') as f:
            orders = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data/few_shots.json'), 'r', encoding='utf-8') as f:
            few_shots = json.load(f)

        return faq, orders, few_shots


    def __call__(self, session_id, save_for_eval=False):
        out_for_eval = []
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logging.info(f"User: {user_text}")

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                continue

            try:
                ai_message = self.chain_with_history.invoke(
                    {"question": user_text, "faq": self.faq, "orders": self.orders,
                     "format_instructions":format_instructions, "few_shots":self.few_shots},
                    {"configurable": {"session_id": session_id}}
                )
                responce = output_parser.invoke(ai_message)
                bot_reply = responce.answer + '\n' + '\n'.join(responce.actions)
                bot_reply = bot_reply.strip()
                print('Бот:', bot_reply, "\n")
                logging.info(f"Bot: {responce.model_dump()}")

                if save_for_eval:
                    out_for_eval.append({'вопрос':user_text, "ответ": responce.model_dump()})
            except openai.APITimeoutError as e:
                print("Бот: [Ошибка] Превышено время ожидания ответа.")
                logging.error(f"Error: {e}")
                continue
            except openai.APIConnectionError as e:
                print("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
                logging.error(f"Error: {e}")
                continue
            except openai.AuthenticationError as e:
                print("Бот: [Ошибка] Проблема с API‑ключом (неавторизовано).")
                logging.error(f"Error: {e}")
                break  # здесь можно завершить, т.к. дальнейшая работа бессмысленна
            except Exception as e:
                print(f"Бот: [Неизвестная ошибка] {e}")
                logging.error(f"Error: {e}")
                continue

        if save_for_eval:
            with open(os.path.join(os.path.dirname(__file__), '..', 'data/eval_prompt.json'), 'w') as f:
                json.dump(out_for_eval, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    logging.info("=== New session ===")
    bot = CliBot()
    bot("user_123", True)