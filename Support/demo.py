from Support.main import main_agent
compiled, config = main_agent()

print("="*60)
question = "(rag) назови изобретателя микроскопа"
print(question)
res = compiled.invoke({'messages': {"role": "human", "content": question}}, config)
print(res["end_answer"])

print("="*60)
question = "когда был изобретен микроскоп, посмотри в www. не используй другую информацию кроме web"
print(question)
res = compiled.invoke({'messages': {"role": "human", "content": question}}, config)
print(res["end_answer"])

print("="*60)
question = "напиши скрипт и выполни его в песочнице. Задача: создай csv и на основе него - график, длин микроорганизмов используя данные: амеба (0,1мм), инфузория (0,3мм), эвглена (0,05мм), трубач (1мм), коловратка (1мм). график сохрани в ../sandbox/g.png"
print(question)
res = compiled.invoke({'messages': {"role": "human", "content": question}}, config)
print(res["end_answer"])

print("="*60)
question = "найди в www сколько экзопланет открыто за все года наблюдений. нужна информация агрегированная по годам."
print(question)
res = compiled.invoke({'messages': {"role": "human", "content": question}}, config)
print(res["end_answer"])

print("="*60)
question = "а теперь сохрани таблицу 'Агрегированная статистика открытых экзопланет по годам, полученную на прошлом шаге' в planet.csv и покажи график. по оси x - года, по оси y - количество открытых планет"
print(question)
res = compiled.invoke({'messages':{"role": "human", "content": question}}, config)
print(res["end_answer"])

