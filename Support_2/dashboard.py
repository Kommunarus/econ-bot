import matplotlib.pyplot as plt
import seaborn as sns

from Support_2.metrics import indicators


counter, df_plot, results, work_time = indicators()

print("\n📊 ЧИСЛО ВЫЗОВОВ КАЖДОГО УЗЛА")
print("=" * 50)
for route, count in counter.most_common():
    print(f"🔹 {route:<10} : {count}")

# График распределения confidence scores


plt.figure(figsize=(10,6))
sns.boxplot(x='Route', y='Confidence', data=df_plot)
plt.title('Распределение Confidence Score по маршрутам')
plt.ylabel('Confidence')
plt.xlabel('Маршрут')
plt.grid(True)
# plt.show()

print("\n📊 АНАЛИЗ CONFIDENCE SCORE ПО МАРШРУТАМ")
print("=" * 50)

# теперь в консоль


# Выводим таблицу
header = f"{'Маршрут':<12} | {'Кол-во':<6} | {'Среднее':<8} | {'Мин.':<6} | {'Макс.':<6}"
print(header)
print("-" * len(header))

for route, stats in results.items():
    print(f"{route:<12} | {stats['count']:<6} | {stats['avg']:<8} | {stats['min']:<6} | {stats['max']:<6}")

print("\n📌 Выводы:")
for route, stats in results.items():
    print(f" • {route}: средний confidence = {stats['avg']} (разброс от {stats['min']} до {stats['max']})")



# Средние времена выполнения по узлам
print("\n📊 СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ ПО УЗЛАМ")
print("=" * 50)


print(work_time.to_markdown())

# Процент успешных/неуспешных обработок