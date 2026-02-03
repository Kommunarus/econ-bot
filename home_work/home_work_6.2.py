import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import glob
from pydantic import BaseModel, Field
from typing import List
import pathlib
import autogen
import ast

dir_files = './sandbox'
SANDBOX_DIR = pathlib.Path(dir_files)
WORK_DIR = pathlib.Path("./workdir")
load_dotenv('my.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

ALLOWED_MODULES = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'datetime',
                   'math', 'json', 'csv', 're'}

FORBIDDEN_NAMES = {'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
                     'eval', 'exec', '__import__', 'pickle', 'shutil', "compile",
                     "open", "input", "__builtins__"}

allowed = ', '.join(ALLOWED_MODULES)
forbidden = ', '.join(FORBIDDEN_NAMES)

checkpointer = MemorySaver()

class CustomMessagesState(MessagesState):
    files: list[str] = []
    files_content: list[str] = []
    code: str = ""
    img_path: list[str] = []
    retry_count: int = 0
    error_message: str = ""
    result: str =""

def node_read_files(state: CustomMessagesState):
    files = glob.glob(dir_files + "/*.csv")
    names_list = [os.path.basename(row) for row in files]
    list_to_str = '\n'.join(names_list)

    content = []
    for file in files:
        with open(file, "r") as f:
            rows = f.readlines()
        if len(rows) == 0:
            content.append("File is empty")
        else:
            content.append(''.join(rows[: min(3, len(rows))]))

    return {'files': names_list, 'files_content': content}


model = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE)

class RouterProfile(BaseModel):
    code: str = Field(description="Код")
    img_path: List[str] = Field(description="Список изображений, которые будут созданы на диске, и на которые есть ссылки в коде")

coder_model = model.with_structured_output(RouterProfile).bind(temperature=0.1)


def node_write_code(state: CustomMessagesState):

    if len(state["files"])>0 and len(state["files_content"])>0 and len(state["files"])==len(state["files_content"]):
        text_about_files = 'Тебе доступны следующие файлы:\n'
        for name, content in zip(state["files"], state["files_content"]):
            text_about_files += f"{name}\nЕго содержимое:\n{content}\n"
    else:
        text_about_files = 'На диске нет файлов.\n'

    CODER_PROMPT = f"""Ты аналитик данных. 
    ТВОЯ ЗАДАЧА:
    Тебе нужно подготавливать Python‑код для выполнения задачи, которую тебя попросят решить.
    Если речь идёт о визуальном отображении, прописывай в коде сохранение графиков с уникальными именами в песочнице 
    и возвращай ссылки пути до файлов.
    Если код не выводит данные для отображения через print, а просто сохраняет изображение, то все равно выводи через print() информацию 
    о том, что сделано.

    КОНТЕКСТ:
    При генерации кода учитывай, что:
    - Есть доступ к записи и чтению файлов в песочнице по адресу '../{SANDBOX_DIR.name}'
    
    - {text_about_files}
    
    КОНТЕКСТ:
    При генерации кода учитывай что:
    - В распоряжении есть такие библиотеки как: {allowed}
    - Запрещено использовать библиотеки: {forbidden}
    
    ФОРМАТ ОТВЕТА:
    code: python код без комментариев и форматирования
    img_path: Путь на диске к создаваемым изображениям. Напиши путь используя '../{SANDBOX_DIR.name}'
    Обязательно перечисляй все новый изображений в img_path.
    """

    if state.get("error_message"):
        user_message = f"""{state['messages'][-1].content}

        ПРЕДЫДУЩАЯ ПОПЫТКА ЗАВЕРШИЛАСЬ ОШИБКОЙ:

        КОД, КОТОРЫЙ ВЫЗВАЛ ОШИБКУ:
        ```python
        {state.get('code', '')}
        ```

        ТЕКСТ ОШИБКИ:
        {state['error_message']}

        ИСПРАВЬ КОД С УЧЁТОМ ЭТОЙ ОШИБКИ."""
    else:
        user_message = state["messages"][-1].content

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=user_message)
    ])
    chain = prompt_template | coder_model
    result = chain.invoke({})
    return {
        "code": result.code,
        "img_path": result.img_path
    }

_executor = None
def get_executor():
    global _executor
    if _executor is None:
        _executor = autogen.UserProxyAgent(
            name="executor",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": str(WORK_DIR), "use_docker": False}
        )
    return _executor

class CodeValidator(ast.NodeVisitor):
    def __init__(self):
        self.error = None

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                self.error = f"❌ Модуль '{root}' не разрешён"
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module is None:
            self.error = "❌ Относительные импорты запрещены"
            return
        root = node.module.split(".")[0]
        if root not in ALLOWED_MODULES:
            self.error = f"❌ Модуль '{root}' не разрешён"
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                self.error = f"❌ Запрещённая функция: {node.func.id}"

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_NAMES:
                self.error = f"❌ Запрещённый атрибут: {node.func.attr}"
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self.error = "❌ Доступ к dunder-атрибутам запрещён"
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            try:
                p = pathlib.Path(node.value)
                if p.is_absolute() or "/" in node.value or "\\" in node.value:
                    resolved = (SANDBOX_DIR / p).resolve()
                    if not resolved.is_relative_to(SANDBOX_DIR.resolve()):
                        self.error = f"❌ Доступ вне sandbox запрещён: {node.value}"
            except Exception:
                pass
        self.generic_visit(node)

def validate_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"❌ SyntaxError: {e}"

    validator = CodeValidator()
    validator.visit(tree)
    if validator.error:
        return False, validator.error
    return True, "✅ Код безопасен для выполнения в sandbox"


def node_run_code(state: CustomMessagesState):
    code = state['code']
    is_safe, message = validate_code(code)
    if not is_safe:
        return {
            "result": "",
            "error_message": message,
            "retry_count": state.get("retry_count", 0) + 1
        }
    try:
        executor = get_executor()
        exit_code, output = executor.execute_code_blocks([("python", code)])

        if exit_code == 0:
            return {
                "result": output,
                "error_message": "",
                "retry_count": state.get("retry_count", 0)
            }
        else:
            return {
                "result": "",
                "error_message": output,
                "retry_count": state.get("retry_count", 0) + 1
            }
    except Exception as e:
        return {
            "result": "",
            "error_message": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }

def should_retry(state: CustomMessagesState, config: RunnableConfig) -> str:
    max_retries = config.get("configurable", {}).get("max_retries", 10)

    if state.get("error_message") and state.get("retry_count", 0) < max_retries:
        return "retry"
    else:
        return "end"


graph = StateGraph(CustomMessagesState)

graph.add_node("read_files", node_read_files)
graph.add_node("write_code", node_write_code)
graph.add_node("run_code", node_run_code)
graph.add_edge(START, "read_files")
graph.add_edge("read_files", "write_code")
graph.add_edge("write_code", "run_code")
graph.add_conditional_edges(
    "run_code",
    should_retry,
    {
        "retry": "write_code",  # Возвращаемся к генерации кода
        "end": END                 # Заканчиваем работу
    }
)


app = graph.compile(interrupt_before=["run_code"], checkpointer=checkpointer,)

# question = ("Прочти файл A.csv и B.csv. Объедини их, и сохрани результат в C.csv. "
#             "Построй график. Ось X - вторая колонка, ось Y - третья")
question = ("прочитай файл train.csv. и построй два отдельных графика."
            "1. по горизонтали - pclass, по вертикали - средний fare"
            "2. график в виде точек, по оси x - колонка age, y - стоимость проезда fare")
input_for_agent = {"messages": [HumanMessage(question)]}

config = {
    "configurable": {
        "thread_id": "code-review-1",
        "max_retries": 10  # Здесь настраиваем лимит попыток
    }
}

result = app.invoke(input_for_agent, config)


while True:
    current_state = app.get_state(config)
    print(f"\n==== Сгенерированный код (попытка {current_state.values.get('retry_count', 0) + 1}) ====")
    print(current_state.values['code'])

    print("\n[OK] Для подтверждения. Либо исправленный код, а потом [+]")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "+":
            user_input = line
            break
        lines.append(line)
    if lines:
        user_input = "\n".join(lines)

    if user_input.strip().upper() != "+":
        # Обновляем код
        app.update_state(config, {"code": user_input})
        updated_state = app.get_state(config)
        print(f"\n==== Обновлённый код ====")
        print(updated_state.values['code'])

    print("\n=== Выполнение кода ===")
    result = app.invoke(None, config)
    print(f"\n==== Попыток выполнено {result.get('retry_count', 0)}====")

    # Проверяем состояние после выполнения
    final_state = app.get_state(config)

    # Если нет ошибки – успех!
    if not final_state.values.get('error_message'):
        print("\n✅ КОД ВЫПОЛНЕН УСПЕШНО!")
        print(f"\n==== result ====")
        print(result['result'])
        print(f"\n==== img_path ====")
        print(result['img_path'])
        break

    # Проверяем лимит попыток (по умолчанию 10)
    max_retries = config.get("configurable", {}).get("max_retries", 10)
    if final_state.values.get('retry_count', 1) >= max_retries:
        print(f"\n⚠️ ВНИМАНИЕ: Достигнут лимит попыток ({max_retries}). Код не выполнен успешно.")
        print(f"\n==== error_message ====")
        print(result['error_message'])
        break

    # Если граф остановился на interrupt_before, значит он готов к новой попытке
    if final_state.next == ('run_code',):
        print(f"\n🔄 Система автоматически перегенерировала код. Проверьте новую версию...")
        continue
    else:
        print(f"\n⚠️ Граф завершился с ошибкой")
        break
        


# """==== Сгенерированный код (попытка 1) ====
# ...
#
# [OK] Для подтверждения. Либо исправленный код, а потом [+]
# -
# +
#
# ==== Обновлённый код ====
# -
#
# === Выполнение кода ===
#
# ==== Попыток выполнено 1====
#
# 🔄 Система автоматически перегенерировала код. Проверьте новую версию...
#
# ==== Сгенерированный код (попытка 2) ====
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Read train.csv from sandbox
# train = pd.read_csv('../sandbox/train.csv', dtype={'age': 'float64', ...})
# ...
#
# [OK] Для подтверждения. Либо исправленный код, а потом [+]
# +
#
# === Выполнение кода ===
#
# ==== Попыток выполнено 2====
#
# 🔄 Система автоматически перегенерировала код. Проверьте новую версию...
#
# ==== Сгенерированный код (попытка 3) ====
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Read train.csv from sandbox
# train = pd.read_csv('../sandbox/train.csv')
#
# # 1. Bar plot: pclass vs average fare
# plt.figure(figsize=(6,4))
# avg_fare_by_pclass = train.groupby('pclass')['fare'].mean()
# sns.barplot(x=avg_fare_by_pclass.index, y=avg_fare_by_pclass.values)
# plt.xlabel('Pclass')
# plt.ylabel('Average Fare')
# plt.title('Average Fare by Pclass
# ')
# plt.tight_layout()
# bar_path = '../sandbox/average_fare_by_pclass.png'
# plt.savefig(bar_path)
# print(f'Bar plot saved to {bar_path}')
#
# # 2. Scatter plot: (x=..???)\n
# """
# We need to produce code only, no comments or formatting. Also must list img_path. Let's craft final answer accordingly.
# code: import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# train = pd.read_csv('../sandbox/train.csv')
#
# plt.figure(figsize=(6,4))
# avg_fare_by_pclass = train.groupby('pclass')['fare'].mean()
# sns.barplot(x=avg_fare_by_pclass.index, y=avg_fare_by_pclass.values)
# plt.xlabel('Pclass')
# plt.ylabel('Average Fare')
# plt.title('Average Fare by Pclass')
# plt.tight_layout()
# bar_path = '../sandbox/average_fare_by_pclass.png'
# plt.savefig(bar_path)
# print(f'Bar plot saved to {bar_path}')
#
# plt.figure(figsize=(6,4))
# sns.scatterplot(data=train, x='age', y='fare')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Fare vs Age')
# plt.tight_layout()
# scatter_path = '../sandbox/fare_vs_age.png'
# plt.savefig(scatter_path)
# print(f'Scatter plot saved to {scatter_path}')
#
# img_path: ../sandbox/average_fare_by_pclass.png, ../sandbox/fare_vs_age.png
#
#
# [OK] Для подтверждения. Либо исправленный код, а потом [+]
# +
#
# === Выполнение кода ===
#
# ==== Попыток выполнено 3====
#
# 🔄 Система автоматически перегенерировала код. Проверьте новую версию...
#
# ==== Сгенерированный код (попытка 4) ====
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# train = pd.read_csv('../sandbox/train.csv')
#
# plt.figure(figsize=(6,4))
# avg_fare_by_pclass = train.groupby('pclass')['fare'].mean()
# sns.barplot(x=avg_fare_by_pclass.index, y=avg_fare_by_pclass.values)
# plt.xlabel('Pclass')
# plt.ylabel('Average Fare')
# plt.title('Average Fare by Pclass')
# plt.tight_layout()
# bar_path = '../sandbox/average_fare_by_pclass.png'
# plt.savefig(bar_path)
# print(f'Bar plot saved to {bar_path}')
#
# plt.figure(figsize=(6,4))
# sns.scatterplot(data=train, x='age', y='fare')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Fare vs Age')
# plt.tight_layout()
# scatter_path = '../sandbox/fare_vs_age.png'
# plt.savefig(scatter_path)
# print(f'Scatter plot saved to {scatter_path}')
#
# [OK] Для подтверждения. Либо исправленный код, а потом [+]
# +
#
# === Выполнение кода ===
#
# >>>>>>>> EXECUTING CODE BLOCK 0 (inferred language is python)...
#
# ==== Попыток выполнено 3====
#
# ✅ КОД ВЫПОЛНЕН УСПЕШНО!
#
# ==== result ====
#
# Bar plot saved to ../sandbox/average_fare_by_pclass.png
# Scatter plot saved to ../sandbox/fare_vs_age.png
#
#
# ==== img_path ====
# ['../sandbox/average_fare_by_pclass.png', '../sandbox/fare_vs_age.png']
# Disconnected from server"""