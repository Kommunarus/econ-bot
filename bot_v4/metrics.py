# count_tokens() — точный подсчёт через tiktoken
# calculate_cost() — расчёт стоимости по ценам модели
# percentile() — вычисление P50, P95, P99
# check_budget() — проверка превышения лимита с BudgetExceeded
# @metrics_model_wrapper — обёртка для LLM вызовов
# @metrics_tool_wrapper — обёртка для инструментов
# print_snapshot() — красивый вывод статистики
# save_metrics_to_file() — сохранение в JSON
# get_route_from_response() — извлечение использованного маршрута из ответа в логи


from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
import time
import logging
import json

logging.basicConfig(
    filename="./logs/chat_session.log",
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
    "output_file": "./logs/metrics.json",
}

# Дополните структуру метрик
metrics = {
    # Общие счётчики
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,

    # Токены и стоимость
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost_usd": 0.0,

    # Производительность
    "latencies": [],  # для расчёта перцентилей

    "routing": {
        # Простые маршруты (один инструмент или без инструментов)
        "simple_routes": {
            "rag_only": 0,
            "web_search_only": 0,
            "order_tracker_only": 0,
            "calculator_only": 0,
            "currency_only": 0,
            "direct_answer": 0,  # без инструментов вообще
        },

        # Комбинированные маршруты (несколько инструментов)
        "complex_routes": {
            "rag_then_web": 0,  # сначала RAG, потом веб
            "rag_with_calculator": 0,  # RAG + вычисления
            "rag_with_currency": 0,  # RAG + конвертация
            "web_with_calculator": 0,
            "multi_tool": 0,  # 3+ инструментов
        },

        # Последовательность вызовов для анализа
        "route_sequences": [],  # ["rag", "calculator"] или ["web_search"] и т.д.

        # Предсказания vs реальность
        "routing_accuracy": {
            "predicted_correct": 0,  # предсказание совпало
            "predicted_incorrect": 0,  # предсказание не совпало
            "fallback_to_llm": 0,  # правило не сработало, передали LLM
        }
    },

    # Использование инструментов
    "tools_errors": {},  # {"tool_name": error_count}

    # Инструменты
    "tools_used": {},  # {"tool_name": count}

    # Ошибки
    "errors": [],  # детальные логи ошибок
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


def get_route_from_tool(tool_name):
    metrics['routing']['simple_routes'][tool_name+'_only'] += 1

def get_route_from_response(response):
    for msg in response.result:
        if isinstance(msg, AIMessage):
            if msg.response_metadata['finish_reason'] == 'tool_calls':
                pass
            else:
                metrics['routing']['simple_routes']['direct_answer'] += 1



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
    calculate_cost('openai/gpt-oss-20b', result)
    check_budget()
    get_route_from_response(response)

    # if metrics['total_calls'] % CONFIG['snapshot_interval'] == 0:
    #     print_snapshot()
    save_metrics_to_file()


    return response


@wrap_tool_call
def metrics_tool_wrapper(request, handler):
    """
    Обёртка для отслеживания инструментов
    """
    tool_name = request.tool_call.get("name", "unknown")
    # if metrics['tools_used'].get(tool_name):
    #     metrics['tools_used'][tool_name] += 1
    # else:
    #     metrics['tools_used'][tool_name] = 1
    get_route_from_tool(tool_name)

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
    # if metrics['tools_used']:
    #     print(f"{COLORS['info']}🛠️  ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ:{COLORS['reset']}")
    #     for name_tool, count_tool in metrics['tools_used'].items():
    #         print(f"  {name_tool:<18} : {count_tool:>10}")
    # print()

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



