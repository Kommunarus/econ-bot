import json
import pandas as pd
import collections

def indicators():
    path_file_log = './logs/support.log'

    logs = []
    with open(path_file_log, 'r') as f:
        text = f.readlines()
        for line in text:
            logs.append(json.loads(line))

    df = pd.DataFrame(logs)

    series1 = df[(df['step'] == 'planner') & (df['input'] != 'Время работы')]['output']

    # Топ-5 самых частых типов запросов. у меня всего 4 типа запроса - раг, веб, дата, ллм, поэтому топа нет.
    plan = []
    for row in series1.values:
        plan.append(row['route'])
    # print('Число вызовов узлов.')
    counter = collections.Counter(plan)

    # вероятность планера в том, что нужно выбрать тот или иной шаг
    data = {}
    for row in series1.values:
        old_v = data.get(row['route'])
        if old_v is not None:
            old_v.append(row['confidence'])
            data[row['route']] = old_v
        else:
            data[row['route']] = [row['confidence']]


    plot_data = [(k, v) for k, vals in data.items() for v in vals]
    df_plot = pd.DataFrame(plot_data, columns=['Route', 'Confidence'])

    # вероятность того, что планер прав в выборе шага, но уже со статистикой
    results = {}
    for route, scores in data.items():
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        count = len(scores)

        results[route] = {
            "avg": round(avg_score, 3),
            "min": min_score,
            "max": max_score,
            "count": count
        }

    # среднее время работы узлов
    pd3 = df[df['input'] == 'Время работы']
    work_time = pd3.filter(items=["step", "output"]).groupby('step').mean()


    return counter, df_plot, results, work_time
