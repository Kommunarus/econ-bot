import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.retriever import hybrid_retriver, check_confidence, format_docs

load_dotenv('.env.example')
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

llm_checker = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    temperature=0.1
)

llm_bot = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    temperature=0.1,
)

querys = [
    'Кем был Омар Хайям?',
    'Какое древнее государство является родиной Омара Хайяма?',
    "Какую религиозную книгу тщательно изучил Омар Хайям?",
    "Какую дисциплину Омар Хайям углублённо изучал с самого детства?",
    "Кем мог работать Хайям после окончания обучения?",
    "Какую должность получил Хайям во дворце султана Мелик-шаха?",
    "Что разработал Омар Хайям во время работы в дворцовой обсерватории?",
    "Большой вклад в развитие какой науки внёс Омар Хайям?",
    "Как называются небольшие четверостишия, которые писал Хайям?",
    "В каком веке литературные произведения Хайяма приобрели большую популярность?",
    "Продолжи: Поймешь, когда пройдешь по всем путям земли:...",
    "Продолжи: Не важно мудрецам, светло или темно,...",
    "Продолжи: Везде Твои силки, Ты в них манишь меня,..."
]

true_answers = [
    'Математик, Философ, Поэт.',
    'Персия',
    "Коран",
    "Математика, астрономия.",
    "Врачом",
    "Духовного наставника",
    "Солнечный календарь",
    "Математики",
    "Рубаи",
    "XIX, XX век",
    "Следы блаженств и бед сливаются вдали, Потом с лица земли добро и зло уходят… Так хочешь, болью стань, а хочешь — исцели.",
    "В аду заночевать, в раю ли — все равно, Как и любовникам: атлас на них, палас ли, Под жаркой головой подушка ли, бревно…",
    "Притом грозишь убить, коль полонишь меня. В силки поймав, убив доверчивую жертву, За что ослушником Ты заклеймишь меня?!"
]


def evaluate_qa_with_llm(query: str, prediction: str, reference: str) -> dict:
    prompt = f"""Ты эксперт по оценке качества ответов.

    Вопрос: {query}
    Ожидаемый ответ: {reference}
    Полученный ответ: {prediction}
    
    Оцени, насколько полученный ответ соответствует ожидаемому по смыслу.
    Ответь ТОЛЬКО одним словом: CORRECT или INCORRECT"""

    response = llm_checker.invoke(prompt)
    verdict = response.content.strip().upper()
    score = 1.0 if verdict == "CORRECT" else 0.0

    return score, verdict



prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Ты знаток творчества Омара Хайяма. "
    "Отвечай по теме творчества Омара Хайяма ТОЛЬКО на основе приведенного контекста. Плюс учитывай историю разговора. "
    "Если тебя просят написать рубайю - пиши ее полностью из контекста. Ничего не выдумывай. "
    "Если тебя спрашивают общие вопросы по творчеству Омара Хайяма - пиши КРАТКА, только по контексту + учитывай "
    "историю сообщений."
    "Если контекста нет - ответь, что не знаешь ответа. Это означает что в базе знаний ничего не нашлось по "
    "запрашиваемой теме. "
    ""),
    ("user", "Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:")
])

check_step = RunnableLambda(check_confidence)
Hybrid_retriever = hybrid_retriver()

retriever_chain = Hybrid_retriever | check_step | format_docs
chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnablePassthrough() | retriever_chain
        }
    | prompt
    | llm_bot
)

predicts = []
for q in querys:
    res =chain.invoke(q)
    predicts.append(res.content)


print("=== Оценка с помощью LLM ===\n")
scores_llm = []
for query, reference, prediction in zip(querys, true_answers, predicts):
    score, verdict = evaluate_qa_with_llm(
        query=query,
        prediction=prediction,
        reference=reference
    )
    scores_llm.append(score)
    print(f"Вопрос: {query}")
    print(f"Полученный ответ: {prediction}")
    print(f"Ожидаемый ответ: {reference}")
    print(f"Оценка: {score} ({verdict})\n")

avg_score_llm = sum(scores_llm) / len(scores_llm)
print(f"Средняя оценка (LLM): {avg_score_llm:.2f}")