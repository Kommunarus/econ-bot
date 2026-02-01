from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import math

class MyMessagesState(MessagesState):
    n: int
    result: int
    final_answer: str

graph = StateGraph(MyMessagesState)

def parse_input(state: MyMessagesState):
    imp = input("Введите число: ")
    try:
        out = int(imp)

        if out <= 20:
            return {'n': out}
        else:
            print("Введите число до 20")
            return {}
    except:
        print("Введите число, а не строку.")
        return {}


def compute_fact(state: MyMessagesState):
    out = math.factorial(state['n'])
    return {'result': out}

def finish(state: MyMessagesState):
    out = f"{state['n']}! = {state['result']}"
    print(out)
    return {'final_answer': out}

def should_continue(state: MyMessagesState):
    if isinstance(state.get('n'), int):
        return "ComputeFact"
    return "ParseInput"


graph.add_node("ParseInput", parse_input)
graph.add_node("ComputeFact", compute_fact)
graph.add_node("Finish", finish)

graph.add_edge(START, "ParseInput")
# graph.add_edge("ParseInput", "ComputeFact")
graph.add_conditional_edges(
    "ParseInput",
    should_continue,
    ["ParseInput", "ComputeFact"]
)
graph.add_edge("ComputeFact", "Finish")
graph.add_edge("Finish", END)


checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

config = {'configurable': {"thread_id": "p1"}}

app.invoke({}, config)


