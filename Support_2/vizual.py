import numpy as np

from Support_2.main import main_agent
from Support_2.metrics import indicators
import networkx as nx
import matplotlib.pyplot as plt
import json

def create_viz_graph():
    compiled_graph, config = main_agent()
    json_code = compiled_graph.get_graph().to_json()
    # print(json.dumps(json_code, indent=4))

    counter, _, _, work_time = indicators()
    new_json = json_code.copy()
    pass

    # Создаем граф NetworkX
    G = nx.DiGraph()

    # Добавляем узлы
    for node in new_json["nodes"]:
        G.add_node(node["id"])

    # Добавляем рёбра с метками
    for edge in new_json["edges"]:
        label = edge.get("condition", "")
        G.add_edge(edge["source"], edge["target"], label=label)

    # Позиционирование узлов
    levels = {
        4: ["__start__"],
        3: ["planner"],
        2: ["RAG", "Web", "Data",],
        1: ["llm"],
        0: ["__end__"]
    }
    pos = nx.multipartite_layout(G, subset_key=levels, align="horizontal")
    pos['RAG'] = np.array([-0.25, 0.0])
    pos['llm'] = np.array([0.25,-0.5])
    pos['__end__'] = np.array([0.25,-1.0])

    # Рисуем граф
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold", arrows=True)

    # Добавляем метки на рёбрах
    edge_labels = nx.get_edge_attributes(G, 'label')
    edge_labels[('planner', 'Data')] = counter.get('DATA', 0)
    edge_labels[('planner', 'RAG')] = counter.get('RAG', 0)
    edge_labels[('planner', 'Web')] = counter.get('WEB_SEARCH', 0)
    edge_labels[('planner', 'llm')] = counter.get('DEFAULT', 0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20)

    # Добавляем время выполнения рядом с узлами
    dict_names = {"Data":"data", "RAG":"rag", "Web":"web", "llm":"llm", "planner":"planner"}
    work_dic = work_time.to_dict()['output']
    for node, (x, y) in pos.items():
        if node in ["Data", "RAG", "Web", "llm", "planner"]:

            time_label = f"{work_dic.get(dict_names[node], 0):.2f}s"
            plt.text(x-0.03, y, time_label, fontsize=12, ha='right', va='center')

    plt.title("LangGraph Visualization")
    plt.show()

if __name__ == "__main__":
    create_viz_graph()