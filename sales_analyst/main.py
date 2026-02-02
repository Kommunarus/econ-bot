from pathlib import Path
from langchain_core.tools import tool
import autogen
import ast
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import logging

SANDBOX_DIR = Path("./sandbox")
WORK_DIR = Path("./workdir")
SANDBOX_DIR.mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename="./logi.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

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
    'socket',       # Сетевые соединения
    'requests',     # HTTP запросы
    'urllib',       # HTTP запросы
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

                # if p.is_absolute() or "/" in node.value or "\\" in node.value:
                #     resolved = (SANDBOX_DIR / p).resolve()
                #     if not resolved.is_relative_to(SANDBOX_DIR):
                #         self.error = f"❌ Доступ вне sandbox запрещён: {node.value}"
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

    print(f"\n{'=' * 60}")
    print("🔧 Выполняю код в песочнице")
    print(f"{'=' * 60}")
    print(f"{message}\n{code}\n{'-' * 60}")

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

@tool('list_sandbox_files')
def list_files() -> str:
    """Показывает список файлов в папке sandbox"""
    try:
        files = list(SANDBOX_DIR.glob("*"))
        if not files:
            return "📂 Папка sandbox пуста"

        file_list = "\n".join([
            f"  - {f.name} ({f.stat().st_size} bytes)"
            for f in files
        ])
        return f"📂 Файлы в sandbox:\n{file_list}"
    except Exception as e:
        return f"❌ Ошибка: {e}"

load_dotenv('.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

agent = create_agent(
    model=ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0),
    tools=[execute_python, list_files],
    system_prompt="""Ты — аналитик данных с доступом к Python песочнице.

                    Инструменты:
                    1. list_sandbox_files - список файлов в sandbox
                    2. execute_python - выполняет Python код

                    ПРАВИЛА:
                    Разрешено: pandas, numpy, matplotlib, seaborn
                    Запрещено: requests, os, subprocess, socket
                    Файлы лежат только в: '../sandbox/', поэтому все пути к файлам дополняй родительской папкой ../sandbox/

                    Примеры:
                      path = '../sandbox/file.csv'
                    - Читать: pd.read_csv('../sandbox/data.csv')
                    - Писать: plt.savefig('../sandbox/plot.png')

                    После выполнения кода объясни результат простым языком."""
)

def interactive_data_analyst():
    """Интерактивный аналитик данных без памяти"""

    print("\n" + "="*80)
    print("💬 Интерактивный аналитик данных")
    print("="*80)

    while True:
        user_input = input("👤 Вы: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("\n👋 До свидания!")
            break

        if not user_input:
            continue

        user_full_input = "Проанализируй файл '../sandbox/sales_2024.csv'. " + user_input
        try:
            # Запускаем агента
            logging.info('Bot: ' + user_input)
            response = agent.invoke({"messages": {"role": "user", "content": user_full_input}})

            # Обновляем историю
            conversation = response["messages"]

            # Выводим ответ (последнее сообщение от ассистента)
            answer = response["messages"][-1].content
            print(f"\n🤖 Агент: {answer}\n")
            logging.info('Bot: ' + answer)

        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")
            logging.error('error: ' + e)

# Запуск
interactive_data_analyst()


