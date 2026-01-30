from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain.tools import tool
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import time
import logging
import json


logging.basicConfig(
    filename="chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

# Заполните цены для моделей (в USD за 1M токенов)
# проверьте актуальные цены и добавьте другие модели
PRICES = {"gpt-4o-mini": {"input": 0.15, "output": 0.60},
          "openai/gpt-oss-20b": {"input": 1.25, "output": 10.00},}

# Настройте параметры
CONFIG = {
    "snapshot_interval": 5,  # каждые N запросов выводить статистику
    "budget_limit_usd": 10.0,  # лимит бюджета
    "save_to_file": True,  # сохранять ли в файл
    "output_file": "metrics.json",
}

# Дополните структуру метрик
metrics = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,

    # Токены
    "input_tokens": 0,
    "output_tokens": 0,

    # Стоимость
    "total_cost_usd": 0.0,

    # Латентность (в секундах)
    "latencies": [],

    # Инструменты
    "tools_used": {},  # {"tool_name": count}

    # Ошибки
    "errors": [],  # [{"type": "...", "message": "...", "timestamp": ...}]
}

def count_tokens(res) -> int:
    metadata = res.usage_metadata
    input_tokens = metadata.get("input_tokens", 0)
    output_tokens = metadata.get("output_tokens", 0)
    metrics['input_tokens'] += input_tokens
    metrics['output_tokens'] += output_tokens

def percentile(p: float) -> float:
    """
    Вычислить перцентиль
    """
    arr = metrics['latencies']
    pr = sorted(arr)[int(p * len(arr))]
    return pr

def calculate_cost(model_name: str, res) -> float:
    prices = PRICES[model_name]
    metadata = res.usage_metadata
    input_tokens = metadata.get("input_tokens", 0)
    output_tokens = metadata.get("output_tokens", 0)
    cost_input = (input_tokens / 1_000_000) * prices["input"]
    cost_output = (output_tokens / 1_000_000) * prices["output"]
    metrics['total_cost_usd'] += (cost_input + cost_output)

def check_budget():
    """
    Проверить, не превышен ли бюджет
    """
    if metrics["total_cost_usd"] >= CONFIG['budget_limit_usd']:
        logging.error("❌ Порог бюджета превышен!")
        raise BudgetExceeded("Превышен лимит бюджета")


class BudgetExceeded(Exception):
    """Исключение при превышении бюджета"""
    pass


@wrap_model_call
def metrics_model_wrapper(request, handler):
    """
    Основная обёртка для сбора метрик модели
    """

    metrics['total_calls'] += 1
    start = time.time()
    try:
        response = handler(request)
        metrics['successful_calls'] += 1
    except Exception as e:
        metrics["failed_calls"] += 1
        metrics["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "timestamp": time.time()
        })


    elapsed = time.time() - start
    metrics['latencies'].append(elapsed)

    result = response.result[0]

    count_tokens(result)
    calculate_cost(MODEL, result)
    check_budget()

    if metrics['total_calls'] % CONFIG['snapshot_interval'] == 0:
        print_snapshot()


    return response


@wrap_tool_call
def metrics_tool_wrapper(request, handler):
    """
    Обёртка для отслеживания инструментов
    """
    tool_name = request.tool_call.get("name", "unknown")
    if metrics['tools_used'].get(tool_name):
        metrics['tools_used'][tool_name] += 1
    else:
        metrics['tools_used'][tool_name] = 1

    result = handler(request)
    return result


def print_snapshot():
    """
    Вывести текущий снимок метрик

    Красиво отформатируйте и выведите:
    - Количество запросов (успешных и с ошибками)
    - Токены (вход + выход)
    - Стоимость (в USD и RUB)
    - Латентность (среднее, P50, P95, P99)
    - Использованные инструменты
    - Процент ошибок
    """
    # Цвета для терминала
    COLORS = {
        'header': '\033[1;36m',  # Голубой жирный
        'success': '\033[0;32m',  # Зеленый
        'warning': '\033[0;33m',  # Желтый
        'error': '\033[0;31m',  # Красный
        'info': '\033[0;37m',  # Белый
        'reset': '\033[0m'  # Сброс цвета
    }

    separator = "=" * 60

    print(f"\n{COLORS['header']}{separator}")
    print("📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
    print(separator + COLORS['reset'])

    # Основные метрики
    print(f"{COLORS['info']}📈 ОСНОВНЫЕ ПОКАЗАТЕЛИ:{COLORS['reset']}")
    print(f"  Всего запросов     : {metrics['total_calls']:>10}")
    print(f"  Успешных           : {COLORS['success']}{metrics['successful_calls']:>10}{COLORS['reset']}")
    print(f"  С ошибками         : {COLORS['error']}{metrics['failed_calls']:>10}{COLORS['reset']}")
    print()

    # Токены
    print(f"{COLORS['info']}🔤 ТОКЕНЫ:{COLORS['reset']}")
    print(f"  Входные            : {metrics['input_tokens']:>10,}")
    print(f"  Выходные           : {metrics['output_tokens']:>10,}")
    print()

    # Стоимость
    print(f"{COLORS['info']}💰 СТОИМОСТЬ:{COLORS['reset']}")
    print(f"  USD                : {metrics['total_cost_usd']:>10.4f}$")
    print(f"  RUB                : {metrics['total_cost_usd'] * 80:>10.2f}₽")
    print()

    # Латентность
    if metrics['latencies']:
        print(f"{COLORS['info']}⏱️  ВРЕМЯ ОТВЕТА (сек):{COLORS['reset']}")
        print(f"  Медиана (p50)      : {percentile(0.50):>10.3f}")
        print(f"  95-й процентиль    : {percentile(0.95):>10.3f}")
        print(f"  99-й процентиль    : {percentile(0.99):>10.3f}")
    print()

    # Инструменты
    if metrics['tools_used']:
        print(f"{COLORS['info']}🛠️  ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ:{COLORS['reset']}")
        for name_tool, count_tool in metrics['tools_used'].items():
            print(f"  {name_tool:<18} : {count_tool:>10}")
    print()

    # Ошибки
    if metrics['errors']:
        print(f"{COLORS['info']}🐛 ОШИБКИ:{COLORS['reset']}")
        error_stats = {}
        total_errors = len(metrics['errors'])

        for error in metrics['errors']:
            error_type = error["type"]
            error_stats[error_type] = error_stats.get(error_type, 0) + 1

        for error_type, count in error_stats.items():
            percentage = (count / total_errors) * 100
            print(f"  {error_type:<18} : {count:>3} ({percentage:.1f}%)")

    print(f"{COLORS['header']}{separator}{COLORS['reset']}\n")


def save_metrics_to_file():
    """
    Сохранить метрики в JSON файл

    """
    if CONFIG["save_to_file"]:
        with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)


# добавьте пару инструментов (можно взять из предыдущих уроков)
@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение с помощью eval"""
    try:
        result = eval(expression)
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def text_analyzer(text: str) -> str:
    """Анализирует текст"""
    return f"Длина: {len(text)} символов, Слов: {len(text.split())}"



# Соберите всё это вместе в агента и протестируйте работоспособность кода
if __name__ == '__main__':
    load_dotenv('my.env')
    MODEL = os.getenv("MODEL")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")


    model = ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.2
    )

    agent = create_agent(
        model=model,
        tools=[calculator, text_analyzer],
        middleware=[metrics_model_wrapper, metrics_tool_wrapper]
    )

    for i in range(20):
        response = agent.invoke({"messages": [{"role": "user", "content": ("Придумай случайную строку и скажи мне ее длину")}]})

    print(response["messages"][-1].content)

    print_snapshot()
    save_metrics_to_file()

    # Пример работы
    
    #"""============================================================
    # 📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ
    # ============================================================
    # 📈 ОСНОВНЫЕ ПОКАЗАТЕЛИ:
    #   Всего запросов     :        143
    #   Успешных           :        143
    #   С ошибками         :          0
    #
    # 🔤 ТОКЕНЫ:
    #   Входные            :     25,341
    #   Выходные           :     10,682
    #
    # 💰 СТОИМОСТЬ:
    #   USD                :     0.1385$
    #   RUB                :      11.08₽
    #
    # ⏱️  ВРЕМЯ ОТВЕТА (сек):
    #   Медиана (p50)      :      0.568
    #   95-й процентиль    :      0.969
    #   99-й процентиль    :      1.376
    #
    # 🛠️  ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ:
    #   text_analyzer      :         23
    #   calculator         :         20
    #
    # ============================================================"""



    # """{
    #     "total_calls": 30,
    #     "successful_calls": 30,
    #     "failed_calls": 0,
    #     "input_tokens": 5275,
    #     "output_tokens": 2173,
    #     "total_cost_usd": 0.028323749999999998,
    #     "latencies": [
    #         0.565342903137207,
    #         0.4434645175933838,
    #         0.36585164070129395,
    #         0.8771374225616455,
    #         0.3719611167907715,
    #         0.6613559722900391,
    #         0.36299705505371094,
    #         0.754875898361206,
    #         0.3723795413970947,
    #         0.5855114459991455,
    #         0.5248761177062988,
    #         0.31168389320373535,
    #         0.5048840045928955,
    #         0.5550916194915771,
    #         0.45508623123168945,
    #         0.3711071014404297,
    #         1.1598353385925293,
    #         0.45343518257141113,
    #         0.47551918029785156,
    #         0.7197794914245605,
    #         0.3726918697357178,
    #         0.8751001358032227,
    #         0.4735255241394043,
    #         0.6945803165435791,
    #         0.36241650581359863,
    #         0.9971117973327637,
    #         0.5746011734008789,
    #         0.7053601741790771,
    #         0.45281147956848145,
    #         0.5526721477508545
    #     ],
    #     "tools_used": {
    #         "calculator": 6,
    #         "text_analyzer": 4
    #     },
    #     "errors": []
    # }"""