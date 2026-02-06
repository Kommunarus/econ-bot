from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import asyncio
import os
from dotenv import load_dotenv

LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", 10))
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str


class Answer(BaseModel):
    answer: str


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}

app = FastAPI(title="LangChain Agent API")

# Раздача статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Главная страница
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")



# Инициализация агента (пример)
def create_agent_instance():
    """
    Создаёт новый экземпляр агента.
    Агент и модель не шарятся между запросами.
    """
    load_dotenv('my.env')
    MODEL = os.getenv("MODEL")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")

    model = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE)
    return create_agent(model=model, checkpointer=InMemorySaver())


@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    """
    Эндпоинт для отправки вопроса агенту.
    Параллелизм ограничен semaphore на уровне LLM.
    """
    logger.info(f"Получен вопрос: {query.question}, session_id: {query.session_id}")
    session_id = query.session_id

    try:
        async with llm_semaphore:
            if session_id not in agents:
                agents[session_id] = create_agent_instance()
            agent = agents[session_id]

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query.question}]},
                {"configurable": {"thread_id": session_id}},
            )

        answer_text = result["messages"][-1].content
        return Answer(answer=answer_text)

    except Exception as e:
        logger.exception("Ошибка при обработке запроса")
        raise HTTPException(
            status_code=500,
            detail="Ошибка обработки запроса"
        )


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy"}
