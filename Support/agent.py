import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain_core.tools import tool
import autogen
import ast

from Support.rag import retr
from Support.web import web_search

SANDBOX_DIR = Path("./sandbox")
WORK_DIR = Path("./workdir")

load_dotenv('.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

# Глобальный executor (создаётся один раз)
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



# Запрещённые модули (ЧЁРНЫЙ СПИСОК)
FORBIDDEN_NAMES = {
    'os',           # Доступ к системе
    'sys',          # Системные функции
    'subprocess',   # Запуск команд
    'requests',  # HTTP запросы
    'urllib',  # HTTP запросы
    'BeautifulSoup'  # HTTP запросы
    'socket',       # Сетевые соединения
    'eval',         # Выполнение строк как кода
    'exec',         # Выполнение строк как кода
    '__import__',   # Динамический импорт
    'pickle',       # Сериализация (может выполнять код)
    'shutil',       # Работа с файлами
}

# Разрешённые модули (БЕЛЫЙ СПИСОК)
ALLOWED_MODULES = {
    'pandas',       # Анализ данных
    'numpy',        # Математика
    'matplotlib',   # Графики
    'seaborn',      # Визуализация
    'scipy',        # Научные вычисления
    'datetime',     # Работа с датами
    'math',         # Математика
    'json',         # JSON
    'csv',          # CSV
    're',           # Регулярные выражения

}

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
                p = Path(node.value)
                # Не допускаем абсолютных путей вообще
                if p.is_absolute():
                    self.error = f"❌ Абсолютные пути запрещены: {node.value}"
                    return

                # Если путь относительный, проверяем его принадлежность к sandbox
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


@tool
def execute_python(code: str) -> str:
    """
    Выполняет Python код в безопасной песочнице.

    Разрешённые библиотеки: pandas, numpy, matplotlib, seaborn.
    Файлы доступны только в '../sandbox/'.

    Args:
        code: Python код для выполнения

    Пример:
        code = '''
        import pandas as pd
        df = pd.read_csv('../sandbox/data.csv')
        print(df.head())
        '''
    """
    is_safe, message = validate_code(code)
    if not is_safe:
        return f"🚨 ОШИБКА БЕЗОПАСНОСТИ: {message}"

    try:
        executor = get_executor()
        result = executor.execute_code_blocks([("python", code)])
        exit_code, output = result[0], result[1]

        if exit_code == 0:
            return output
        else:
            return f"❌ Ошибка выполнения:\n{output}"
    except Exception as e:
        return f"❌ Исключение: {str(e)}"


@tool
def tool_web_search(query: str, k: int = 10) -> str:
    """Выполняет веб-поиск. Возвращает top-k результатов
    query - запрос агента, возможно измененный и дополненный относительно того, что спрашивал пользователь.
    k - сколько ссылок нужно возвращать по запросу query"""
    return (web_search(query, k))


@tool
def tool_rag(query: str) -> str:
    """Выполняет поиск по внутренним документам в векторой базе faiss.
    query - запрос, будет преобразован в эмбеддинг и по этому эмбеддингу в векторной базе будут искаться совпадения"""
    return retr(query)



def create_code_agent():
    agent = create_agent(
        model=ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0),
        tools=[execute_python],
        system_prompt="""Ты — аналитик данных с доступом к Python песочнице.
    
                        Инструмент: execute_python - выполняет Python код
                        
                        ПРАВИЛА:
                        Разрешено: pandas, numpy, matplotlib, seaborn, scipy
                        Запрещено: os, subprocess, socket
                        Файлы только в: '../sandbox/'
                        
                        Примеры:
                        - Читать: pd.read_csv('../sandbox/data.csv')
                        - Писать: plt.savefig('../sandbox/plot.png')
                        Составь код на python  и вызови инструмент для его выполнения.
                        После выполнения кода объясни результат простым языком."""
    )

    return agent


def create_smart_agent():
    agent = create_agent(
        model=ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0),
        tools=[execute_python, tool_web_search, tool_rag],
        system_prompt="""Ты — аналитик данных с доступом к Python песочнице.

                        Инструменты: 
                        1. execute_python - выполняет Python код

                        ПРАВИЛА для execute_python:
                        Разрешено: pandas, numpy, matplotlib, seaborn, scipy
                        Запрещено: os, subprocess, socket
                        Файлы только в: '../sandbox/'

                        Примеры:
                        - Читать: pd.read_csv('../sandbox/data.csv')
                        - Писать: plt.savefig('../sandbox/plot.png')

                        
                        
                        2. tool_web_search - поиск общей информации и общеизвестных фактов в интернете. Использовать, если запрос прямо не указывает искать в faiss
                        3. tool_rag - поиск наиболее релевантных документов во внутренней документации в базе faiss.
                        
                        Если в execute_python требуется получить доступ в интернет, сначала воспользуйся инструментом tool_web_search, и потом обработай его.
                        Отвечай кратко и по делу.
                        """
    )

    return agent