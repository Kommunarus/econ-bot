from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import asyncio
import os
import json
import uvicorn
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

    model = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, streaming=True)
    return create_agent(model=model, checkpointer=InMemorySaver())




@app.post("/ask")
def ask_stream(query: Query):
    """
    Потоковый эндпоинт с использованием Server-Sent Events (SSE)
    """
    logger.info(f"Получен вопрос: {query.question}, session_id: {query.session_id}")
    session_id = query.session_id

    def generate_response():
        try:
            input_data = {"messages": [{"role": "user", "content": query.question}]}

            if session_id not in agents:
                agents[session_id] = create_agent_instance()
            agent = agents[session_id]

            # stream_mode="messages" — стриминг именно токенов ответа
            for token, metadata in agent.stream(input_data,
                                                config={"configurable": {"thread_id": session_id}},
                                                stream_mode="messages", ):
                # content_blocks — структурированный вывод
                blocks = token.content_blocks

                # Отфильтровываем события инструментов
                if not blocks or hasattr(token, "tool_call_id"):
                    continue

                block = blocks[0]
                if block.get("type") == "text":
                    event = {
                        "type": "token",
                        "content": block["text"]
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Сигнал завершения
            yield 'data: {"type": "done"}\n\n'

        except Exception as e:
            logger.error("Ошибка генерации", exc_info=True)
            error_event = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)