from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from Support.rag import retr
from Support.web import web_search
from Support.agent import create_code_agent, create_smart_agent

load_dotenv('.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
model = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0)

code_agent = create_code_agent()
smart_agent = create_smart_agent()

class CustomMessagesState(MessagesState):
    next_node: str
    query: str
    end_answer: str


def planner(CustomMessagesState):
    last_message = [mesg.content for mesg in CustomMessagesState["messages"]]
    # content = '\n\n'.join(last_message)
    content = last_message[-1]

    is_Data = any(word in content for word in ["данные", "график", "csv", "таблица", "файл"])
    is_Web = any(word in content for word in ["когда", "сколько", "кто", "http", "www", "интернет", "найти", "web"])
    is_RAG = any(word in content for word in ["документ","внутренний", "документации", "rag"])
    if sum([is_Data, is_Web, is_RAG]) == 1:
        if is_Data:
            return {"next_node": "Data",}
        elif is_Web:
            return {"next_node": "Web", "query": content}
        elif is_RAG:
            return {"next_node": "RAG", "query": content}
    else:
        return {"next_node": "Smart_llm",}


def route_next(state):
    next_node = state.get("next_node", "Smart_llm")
    print(next_node)
    return  next_node

def rag(CustomMessagesState):
    out = retr(CustomMessagesState["query"])
    return {"messages": [AIMessage(content=out)]}

def Web(CustomMessagesState):
    out = web_search(CustomMessagesState["query"])
    return {"messages": [AIMessage(content=out)]}

def Data(CustomMessagesState):
    response = code_agent.invoke({"messages": CustomMessagesState["messages"]})
    answer = response["messages"][-1].content
    return {"messages": [AIMessage(content=answer)]}

def SmartLLM(CustomMessagesState):
    response = smart_agent.invoke({"messages": CustomMessagesState["messages"]})
    answer = response["messages"][-1].content
    return {"end_answer": answer, "messages": [AIMessage(content=answer)]}

def llm(CustomMessagesState):
    context_parts = [mesg.content for mesg in CustomMessagesState["messages"]]
    full_context = "\n".join(context_parts) if context_parts else "Нет доступного контекста."

    chat_prompts = ChatPromptTemplate.from_template("Ты консультант магазина микроскопов. Твоя задача отвечать на вопросы по микроскопии.\n"
                                                    "Если тебе дается контекст, используй в ответе только его и предоставь ссылки на источники (название источника и номер страницы, или адрес ссылки). "
                                                    "Не придумывай ничего от себя, только подготовь ответ. Отвечай кратко и по делу.\n"
                                                    "User {query}. \n"
                                                    "Контекст:\n {context}. \n\n"
                                                    "Ответ:")
    prompt = chat_prompts.invoke({"query": CustomMessagesState["query"], "context": full_context})
    res = model.invoke(prompt)
    return {"end_answer": res.content, "messages": [AIMessage(content=res.content)]}



def main_agent():
    graph = StateGraph(CustomMessagesState)
    checkpointer = MemorySaver()


    graph.add_node("planner", planner)
    graph.add_node("RAG", rag)
    graph.add_node("Web", Web)
    graph.add_node("Data", Data)
    graph.add_node("Smart_llm", SmartLLM)
    graph.add_node("llm", llm)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_next, ["RAG", "Data", "Web", "Smart_llm"])
    graph.add_edge("RAG", "llm")
    graph.add_edge("Data", "llm")
    graph.add_edge("Web", "llm")
    graph.add_edge("Smart_llm", END)
    graph.add_edge("llm", END)

    compiled = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "llm_calls-1"}}

    return compiled, config

def main(question):
    compiled, config = main_agent()
    res = compiled.invoke({'messages': {"role": "human", "content": question}}, config)
    print(res["end_answer"])

if __name__ == "__main__":
    # q = "поищи в www информацию о микроскопе микромед 3u3 и запиши/перезапиши эти сайты в micromed.csv файл."
    question = "напиши скрипт и выполни его в песочнице. сообщи о результатах. Задача: покажи csv и на основе него график в виде plot длин микроорганизмов используя данные: амеба (0,1мм), инфузория (0,3мм), эвглена (0,05мм), трубач (1мм), коловратка (1мм). график сохрани в ../sandbox/g.png"
    main(question)