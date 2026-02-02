import ast
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool



load_dotenv('my.env')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
# llm = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0.1)


class Plan(BaseModel):
    steps: list[str] = Field(description="Список шагов решения задачи")


def make_plan(task: str) -> list[str]:
    # Планировщик
    planer_llm = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE,
                            temperature=0.1).with_structured_output(
        schema=Plan,
        method='json_schema'
    )

    planer_template = ChatPromptTemplate.from_messages([
        ("system", ("Ты планировщик. "
                    "Сделайплан шагов, необходимый для ответа на вопрос. "
                    "Сократи план до минимально необходимого, без лишних шагов. "
                    "Каждый шаг должен быть конкретным: "
                    "что-то умножить, что-то сложить, что-то прочитать, и так далее. "
                    "Верни список шагов на русском языке. Не давай финального ответа")),
        ("user", "{task}")
    ])

    planer = planer_template | planer_llm
    plan = planer.invoke({"task": task})

    result_plan = []
    # print("=== ПЛАН ===")
    for i, step in enumerate(plan.steps, 1):
        # print(f"{i}. {step}")
        result_plan.append(step)
    # print()

    return result_plan


@tool("calculator")
def calculator(sample: str) -> str:
    """Инструмент выполняет арифметические выражения.
    Разрешены только базовые операции и функции из math
    Пример: sample='(6+9/2)**2'"""
    import math
    tree = ast.parse(sample, mode='eval')

    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith('_')
    }
    allowed_names.update({
        'abs': abs,
        'max': max,
        'min': min,
        'pow': pow,
        'round': round,
    })

    code = compile(tree, filename='', mode='eval')
    try:
        answer = eval(code, {"__builtins__": {}}, allowed_names)
    except Exception as f:
        print('error calc: ', f)
        answer = f
    return str(answer)


def execute_plan(steps: list[str]) -> list[str]:
    system_prompt = """
    Ты агент-помощник. Тебе нужно выполнять задание пошагово. Тебе будут давать шаги и контекст выполнения предыдущих шагов.
    У тебя есть следующие инструменты:
    calculator: продвинутый калькулятор. Ввод: числовое выражение. Вывод: результат вычисления
    """

    agent = create_agent(
        model=ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0),
        system_prompt=system_prompt,
        tools=[calculator,],
    )

    results = []
    for i, step in enumerate(steps, 1):
        # print(f"Шаг {i}: {step}")

        # Формируем запрос с контекстом
        context_str = "\n".join([f"Результат шага {i}: {r}" for i, r in enumerate(results, 1)])

        if context_str:
            input_text = f"Выполни следующее: {step}\n\nКонтекст предыдущих шагов:\n{context_str}"
        else:
            input_text = f"Выполни следующее: {step}"

        # Выполняем шаг через агента
        result = agent.invoke({"messages": [{"role": "user", "content": input_text}]})
        results.append(result['messages'][-1].content)
        # print(result['messages'][-1].content)
    return results

def make_final_answer(task: str, results: list[str]) -> str:
    final_llm = ChatOpenAI(model_name=MODEL, openai_api_key=API_KEY, openai_api_base=API_BASE, temperature=0)
    final_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты помощник. На основе результатов выполненных шагов дай финальный ответ на исходный вопрос пользователя."),
        ("user", "Исходный вопрос: {task}\n\nРезультаты выполнения:\n{results}\n\nДай краткий финальный ответ.")
    ])

    final_chain = final_prompt | final_llm
    final_response = final_chain.invoke({
        "task": task,
        "results": "\n\n".join([f"Шаг {i + 1}: {r}" for i, r in enumerate(results)])
    })

    # print(final_response.content)
    return final_response.content


def main(task: str):
    plan = make_plan(task)
    ex = execute_plan(plan)
    out = make_final_answer(task, ex)
    return out, plan, ex


task = (""
        "Первый мальчик весит Х кг. "
        "Второй в полтора раза больше, чем первый. "
        "Третий мальчик весит на 10кг меньше, чем второй."
        "Сколько весят все мальчики вместе? "
        "при Х = 30")


res, plan, execution = main(task)
print('='*60)
print(res)
print('-'*60)
print(plan)
print('-'*60)
print(execution)
print('='*60)



# ============================================================
# Суммарный вес всех троих мальчиков равен 110 кг.
# ------------------------------------------------------------
# ['Считать значение X=30', 'Вычислить массу второго мальчика как 1.5*X', 'Вычислить массу третьего мальчика как масса второго -10', 'Сложить массы всех трёх мальчиков']
# ------------------------------------------------------------
# ['X = 30', 'Масса второго мальчика: **45** (единиц массы).', 'Масса третьего мальчика: **35** единиц массы.', 'Сумма масс всех трёх мальчиков: **110** единиц массы.']
# ============================================================