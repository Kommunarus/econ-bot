from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from collections import OrderedDict
from langgraph.checkpoint.base import BaseCheckpointSaver

from Support_2.rag import retr
from Support_2.web import web_search
from Support_2.agent import create_data_agent

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import os
from dotenv import load_dotenv
import time
import json


load_dotenv('.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
model = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0)
data_app, config_data = create_data_agent()



class CustomMessagesState(MessagesState):
    context: str
    route: Literal["RAG", "WEB_SEARCH", "DATA", "DEFAULT"]
    confidence: float
    reasoning: str
    logs: List[Dict[str, Any]] = []

class RouterProfile(BaseModel):
    route: Literal["RAG", "WEB_SEARCH", "DATA", "DEFAULT"] = Field(description="Источник данных для ответа")
    confidence: float = Field(ge=0.0, le=1.0, description="уверенность выбора")
    reasoning: str = Field(description="объяснение выбора")

router_model = model.with_structured_output(RouterProfile).bind(temperature=0.1)
ROUTER_SYSTEM_PROMPT = """Маршрутизируй запрос к одному из источников:

RAG — база знаний компании (документация, FAQ, продукты)
WEB_SEARCH — актуальная информация из интернета
DATA — анализ данных, графики, CSV‑файлы
DEFAULT — общие вопросы без специальных инструментов

Правила:
1. Ключевые слова для DATA: график, диаграмма, построй, csv, таблица
2. Для WEB_SEARCH: новости, актуальное, сейчас, погода, внешние организации
3. Для RAG: "ваш", "вашей компании", внутренняя информация, "у вас"
4. DEFAULT для всего остального

Формат: {"route": "RAG|WEB_SEARCH|DATA|DEFAULT", "confidence": 0.0-1.0, "reasoning": "краткое объяснение"}"""

def log_step(logs: List[Dict], step_name: str, input_data=None, output_data=None):
    logs.append({
        "step": step_name,
        "input": input_data,
        "output": output_data
    })

def log_save(logs: List[Dict]):
    log_files = './logs/support.log'
    with open(log_files, 'a') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False)+'\n')

def planner(state: CustomMessagesState):
    time_s = time.time()

    query = state["messages"][-1].content

    messages=[
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    response = router_model.invoke(messages)

    log_step(state["logs"], "planner", input_data=query, output_data=response.model_dump())
    log_step(state["logs"], "planner", input_data='Время работы', output_data=time.time()-time_s)

    return {
        "route": response.route,
        "confidence": response.confidence,
        "reasoning": response.reasoning
    }


def rag(state: CustomMessagesState):
    time_s = time.time()

    query = state["messages"][-1].content
    context = retr(query)

    log_step(state["logs"], "rag", input_data='Время работы', output_data=time.time()-time_s)

    return {"context": context}

def Web(state: CustomMessagesState):
    time_s = time.time()

    query = state["messages"][-1].content
    context = web_search(query)

    log_step(state["logs"], "web", input_data='Время работы', output_data=time.time()-time_s)


    return {"context": context}

def Data(state: CustomMessagesState):
    time_s = time.time()
    max_retries = config_data["configurable"]["max_retries"]
    # создание кода
    data_app.invoke({
        "messages": state["messages"][-1].content,
        "code": "",
        "result": "",
        "error_message": ""
    }, config_data)

    while True:
        current_state = data_app.get_state(config_data)
        if not current_state.values.get('code'):
            log_step(state["logs"], "data", input_data='Время работы', output_data=time.time() - time_s)
            return {"context": "Не удалось сгенерировать код для анализа данных."}

        # print("\n=== Выполнение кода ===")
        data_app.invoke(None, config_data)
        final_state = data_app.get_state(config_data)

        if not final_state.values.get('error_message'):
            log_step(state["logs"], "data", input_data='Время работы', output_data=time.time() - time_s)
            return {
                "context": final_state.values.get('result', ''),
            }

        if final_state.values.get('retry_count', 0) >= max_retries:
            print(f"\n⚠️ ВНИМАНИЕ: Достигнут лимит попыток ({max_retries}). Код не выполнен успешно.")
            log_step(state["logs"], "data", input_data='Время работы', output_data=time.time() - time_s)
            return {
                "context": f"Не удалось выполнить код после {max_retries} попыток. Последняя ошибка: {final_state.values.get('error_message')}",
            }
        # повторное создание кода, с добавленой ошибкой в промпт
        data_app.invoke(None, config_data)


def generate_response_node(state: CustomMessagesState):
    time_s = time.time()

    query = state["messages"][-1].content
    context = state.get("context", "")
    system_prompt = "Ты — полезный AI‑ассистент. Отвечай максимально кратко."

    if state["route"] in ("RAG", "WEB_SEARCH"):
        system_prompt += ' В контексте тебе дано найденное содержание'
        answer = f'Вопрос: {query}\n\nКонтекст:{context}'
        state["messages"][-1] = answer
    elif state["route"] == "DATA":
        system_prompt += ' В контексте тебе дан вывод кода'
        answer = f'Вопрос: {query}\n\nКонтекст:{context}'
        state["messages"][-1] = answer

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)

    log_step(state["logs"], "llm", input_data='Usage_metadata',
             output_data={key: response.usage_metadata.get(key) for key in ['input_tokens',
                                                                            'output_tokens',
                                                                            'total_tokens']})
    log_step(state["logs"], "llm", input_data='Время работы', output_data=time.time()-time_s)

    log_save(state["logs"])
    state["logs"] = []


    return {"messages": [response]}



def main_agent():
    graph = StateGraph(CustomMessagesState)



    graph.add_node("planner", planner)
    graph.add_node("RAG", rag)
    graph.add_node("Web", Web)
    graph.add_node("Data", Data)
    graph.add_node("llm", generate_response_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: state["route"],
        {
            "RAG": "RAG",
            "WEB_SEARCH": "Web",
            "DATA": "Data",
            "DEFAULT": "llm"
        }
    )
    graph.add_edge("RAG", "llm")
    graph.add_edge("Data", "llm")
    graph.add_edge("Web", "llm")
    graph.add_edge("llm", END)

    compiled = graph.compile()

    config = {"configurable": {"thread_id": "llm_calls-1"}}

    return compiled, config

def main(question):
    compiled, config = main_agent()
    response = compiled.invoke({
        'messages': {"role": "human", "content": question},
        'logs': []
         },
        config=config,
    )
    answer = response["messages"][-1].content
    print(f"\n🤖 Агент: (from {response["route"]}) {answer}\n")

    # current_log = json.dumps(response["logs"], indent=4, ensure_ascii=False)
    # print(f'\nLogs: {current_log}')


def interactive_data_analyst():
    compiled, config = main_agent()
    while True:
        user_input = input("👤 Вы: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("\n👋 До свидания!")
            break

        if not user_input:
            continue


        try:
            response = compiled.invoke(
                {"messages": [HumanMessage(content=user_input)], "logs": []},
                config
            )

            answer = response["messages"][-1].content
            print(f"\n🤖 Агент (from {response["route"]}): {answer}\n")

        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")

if __name__ == "__main__":

    interactive_data_analyst()