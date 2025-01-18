# -*- coding: windows-1251 -*-

import re
from langchain.schema import AIMessage
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAIEmbeddings

from sqlalchemy.sql import text
from starlette.responses import HTMLResponse

from customer.config import openai_api_key
from db import SessionLocal, get_db
from models import PredefinedQuestion  # Импортируем модель взаимодействий
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from customer.config import SessionLocal, logger
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from customer.classification import is_product_related
from customer.faiss import find_product_in_faiss
from models import Interaction
from customer.config import logger
from sqlalchemy.orm import Session
from customer.classification import classify_question, determine_stage
from customer.faiss import save_question_index
import faiss
import numpy as np
from langchain.memory import ConversationBufferMemory
from customer.get_template import find_question_template
from langchain.memory import ConversationBufferWindowMemory

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Создание роутера для чата с клиентами
customer_chat_router = APIRouter()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# Другие необходимые библиотеки
import pandas as pd
import json
import logging
from dotenv import load_dotenv
import spacy

# Pydantic для валидации данных
from pydantic import BaseModel, Field
from typing import Optional

import os
from langchain.schema import HumanMessage

# Функция для очистки HTML-тегов из текста
def clean_html(text):
    logger.debug(f"Очищаем текст от HTML-тегов: {text[:100]}...")  # Логируем начало текста
    clean_text = re.sub(r'<.*?>', '', text)  # Удаляем все HTML теги
    logger.debug(f"Очищенный текст: {clean_text[:100]}...")  # Логируем начало очищенного текста
    return clean_text

# Загрузка переменных окружения
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API ключ OpenAI не найден в переменных окружения")

# Загрузка модели для русского языка
nlp = spacy.load('ru_core_news_sm')

# Инициализация моделей OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)
llm2 = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# Модель для хранения информации о заказе
class OrderDetails(BaseModel):
    """Информация о заказе, извлеченная из запроса пользователя."""
    quantity: Optional[int] = Field(default=None, description="Количество товаров, которые пользователь хочет заказать")
    method: Optional[str] = Field(default=None, description="Метод получения заказа: доставка или самовывоз")


# Проверка существования CSV файла и его создание, если он отсутствует
csv_file = "product_converted.csv"
if not os.path.exists(csv_file):
    df = pd.read_excel("product.xls")
    df.to_csv(csv_file, index=False)

# Создание агента для работы с CSV файлом
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)

# templates = Jinja2Templates(directory="static")

@customer_chat_router.get("/customer_chat")
async def customer_chat(request: Request):
    return templates.TemplateResponse("customer_chat.html", {"request": request})

# Пример создания цепочки для извлечения данных
extraction_chain = (
    PromptTemplate(input_variables=["message"], template="""Извлеки данные в формате JSON:
    - product_name: название товара.
    - quantity: количество.
    - method: доставка/самовывоз.
    Сообщение: {message}""")
    | ChatOpenAI(api_key=openai_api_key)  # Используйте свою модель OpenAI
)

def extract_data(message):
    """
    Извлекает данные из сообщения пользователя.

    Args:
        message (str): Сообщение пользователя.

    Returns:
        dict: Извлечённые данные в формате JSON.

    Raises:
        ValueError: Если результат имеет неожиданный формат.
        Exception: Для любых других ошибок.
    """
    try:
        logger.info(f"Начало извлечения информации из сообщения: {message}")

        try:
            # Шаг 1: Запуск цепочки извлечения
            extraction_result = extraction_chain.invoke({"message": message})

            # Логирование результата
            logger.info(f"Сырой результат извлечения: {extraction_result}")
            logger.info(f"Тип результата извлечения: {type(extraction_result)}")
        except Exception as e:
            logger.error(f"Ошибка при вызове цепочки извлечения: {str(e)}")
            raise ValueError("Ошибка при вызове цепочки извлечения данных.")

        try:
            # Шаг 2: Обработка результата
            extraction_text = None
            if isinstance(extraction_result, AIMessage):
                extraction_text = extraction_result.content
                logger.debug(f"Результат извлечён из AIMessage: {extraction_text}")
            elif isinstance(extraction_result, list) and len(extraction_result) > 0:
                extraction_text = extraction_result[0].content
                logger.debug(f"Результат извлечён из списка: {extraction_text}")
            elif isinstance(extraction_result, str):
                extraction_text = extraction_result
                logger.debug(f"Результат извлечён из строки: {extraction_text}")
            else:
                logger.error(f"Неожиданный формат результата извлечения: {type(extraction_result)}")
                raise ValueError(f"Неожиданный формат результата извлечения: {type(extraction_result)}")

            # Проверка, пуст ли результат
            if not extraction_text:
                logger.warning("Результат извлечения пустой.")
                raise ValueError("Результат извлечения не содержит данных.")
        except Exception as e:
            logger.error(f"Ошибка обработки результата извлечения: {str(e)}")
            raise ValueError("Ошибка обработки результата извлечения данных.")

        try:
            # Шаг 3: Преобразование в JSON
            extracted_data = json.loads(extraction_text.replace("'", '"'))
            logger.info(f"Извлечённые данные в формате JSON: {extracted_data}")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON: {e}")
            raise ValueError("Не удалось преобразовать результат извлечения в JSON.")

    except Exception as e:
        logger.error(f"Ошибка при извлечении данных: {str(e)}")
        raise e

def save_interaction(session: Session, user_id: str, query: str, response: str) -> None:
    """
    Сохраняет взаимодействие в базу данных с использованием SQLAlchemy.

    :param session: SQLAlchemy сессия для выполнения запросов.
    :param user_id: Идентификатор пользователя.
    :param query: Запрос пользователя.
    :param response: Ответ, который будет сохранён.
    """
    try:
        logger.debug(f"Сохраняем взаимодействие для user_id={user_id} с запросом: {query}")

        # Очистка ответа от HTML-тегов
        response_cleaned = clean_html(response)

        # Логируем очищенный ответ перед сохранением
        logger.debug(f"Очищенный ответ: {response_cleaned[:100]}...")  # Логируем начало очищенного ответа

        # Создаём объект взаимодействия
        interaction = Interaction(user_id=user_id, query=query, response=response_cleaned)

        # Добавляем и сохраняем в базе данных
        session.add(interaction)
        session.commit()

        logger.info(f"Взаимодействие успешно сохранено для user_id={user_id}")

    except Exception as e:
        session.rollback()  # Откатываем изменения в случае ошибки
        logger.error(f"Ошибка записи взаимодействия: {e}")


def get_message_history(user_id):
    """
    Извлекает историю сообщений пользователя из базы данных.
    """
    try:
        # Извлекаем все взаимодействия для пользователя, сортируя по времени
        interactions = Interaction.query.filter(Interaction.user_id == user_id).order_by(Interaction.timestamp).all()

        # Формируем историю сообщений
        history = []
        for interaction in interactions:
            history.append({
                "user_message": interaction.query,  # Вопрос пользователя
                "response": interaction.response  # Ответ
            })

        # Логируем количество извлеченных взаимодействий
        logger.info(f"Извлечено {len(history)} взаимодействий для user_id={user_id}")

        # Логируем каждое взаимодействие по отдельности
        for idx, interaction in enumerate(history):
            logger.info(f"Взаимодействие #{idx + 1}: Вопрос - {interaction['user_message']}, Ответ - {interaction['response']}")

        return history

    except Exception as e:
        # Логируем ошибку, если что-то пошло не так
        logger.error(f"Ошибка извлечения истории сообщений: {e}")
        return []


# Определяем тип состояния
class State(TypedDict):
    messages: Annotated[list, add_messages]  # История сообщений
    current_product: dict  # Информация о текущем обсуждаемом товаре


def get_user_state(user_id):
    """Возвращает состояние пользователя или создаёт новое, если его нет."""
    if user_id not in user_states:
        user_states[user_id] = {"messages": [], "current_product": None}
    return user_states[user_id]

def classify_message_with_openai(message: str) -> bool:
    """
    Определяет, связан ли вопрос с товаром, используя OpenAI.
    """
    prompt = f"""
    Вопрос: "{message}"
    Ответьте "да", если вопрос связан с товаром (например, наличием, ценой, характеристиками), 
    или "нет" в противном случае.
    """
    response = llm.invoke(prompt)
    return "да" in response.content.lower()


#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Инициализация памяти для каждого пользователя
user_memories = {}

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)  # Хранить последние 10 сообщений
    return user_memories[user_id]


# Функция-узел графа
async def chatbot(state: State, session: Session = Depends(get_db)) -> State:
    """
    Функция-узел графа для обработки сообщения клиента с учетом стадии и категории.
    """
    try:
        logger.info(f"Начало обработки. Текущее состояние: {state}")

        # Извлечение последнего сообщения пользователя
        messages = state.get("messages", [])
        last_user_message = messages[-1].content if messages else "Привет, бот!"
        logger.info(f"Последнее сообщение пользователя: {last_user_message}")

        # Получение памяти пользователя
        user_memory = get_user_memory(state.get("user_id", "default_user"))

        # Сохраняем сообщение пользователя в память
        user_memory.chat_memory.add_user_message(last_user_message)
        logger.info("Сообщение пользователя сохранено в память.")

        # Проверяем, связан ли вопрос с товаром
        logger.info(f"Проверяем, связан ли вопрос с товаром: '{last_user_message}'")
        is_related = is_product_related(last_user_message)
        logger.info(f"Результат классификации: {'Да' if is_related else 'Нет'} для вопроса: '{last_user_message}'")

        if is_related:
            logger.info("Вопрос связан с товаром.")
            # Попытка найти шаблон ответа через FAISS
            template = find_question_template(last_user_message)
            current_product = state.get("current_product")

            if template:
                logger.info(f"Найден шаблон ответа через FAISS: {template}")

                # Если информации о товаре нет, ищем товар через FAISS
                if not current_product:
                    logger.info("Текущая информация о товаре отсутствует. Пытаемся найти товар через FAISS.")
                    current_product = find_product_in_faiss(last_user_message)
                    if current_product:
                        state["current_product"] = current_product
                        logger.info(f"Информация о найденном товаре добавлена в состояние: {current_product}")

                # Формируем ответ, подставляя данные о товаре в шаблон
                if current_product and all(key in current_product for key in ["product_name", "availability", "price"]):
                    response_text = template.format(
                        product_name=current_product["product_name"],
                        availability=current_product["availability"],
                        price=current_product["price"]
                    )
                    # Сохраняем информацию о товаре в память
                    memory_message = (
                        f"Обсуждаем товар: {current_product['product_name']}. "
                        f"Доступно: {current_product['availability']} шт. Цена: {current_product['price']} руб."
                    )
                    user_memory.chat_memory.add_ai_message(memory_message)
                    logger.info(f"Информация о товаре сохранена в память: {memory_message}")
                else:
                    logger.warning("Информация о товаре некорректна или отсутствует. Используем дефолтные значения.")
                    response_text = template.format(
                        product_name="неизвестный товар",
                        availability="неизвестно",
                        price="неизвестно"
                    )
            else:
                logger.warning("Шаблон не найден через FAISS. Выполняется поиск товара.")
                product_info = find_product_in_faiss(last_user_message)

                if product_info:
                    logger.info(f"Найден товар: {product_info}")
                    state["current_product"] = product_info
                    response_text = (
                        f"{product_info['product_name']}' в наличии: "
                        f"{product_info['availability']} шт. Стоит: {product_info['price']} руб. "
                        f"Сколько вам нужно?"
                    )
                    memory_message = (
                        f"Обсуждаем товар: {product_info['product_name']}. "
                        f"Доступно: {product_info['availability']} шт. Цена: {product_info['price']} руб."
                    )
                    user_memory.chat_memory.add_ai_message(memory_message)
                    logger.info(f"Информация о найденном товаре сохранена в память: {memory_message}")
                else:
                    logger.warning("Товар не найден.")
                    response_text = "Одну минуту, уточню"
        else:
            logger.info("Вопрос не связан с товаром. Генерация общего ответа через нейросеть.")
            # Проверяем, есть ли сохраненные данные о товаре
            if state.get("current_product"):
                product_info = state["current_product"]
                logger.info("Используется информация о товаре из состояния.")

                # Передача информации о товаре в модель для формирования ответа
                llm = ChatOpenAI(api_key=openai_api_key)
                product_context_template = PromptTemplate.from_template(
                    """
                    Учитывая следующую информацию о товаре:
                    Название: {product_name}
                    В наличии: {availability} шт.
                    Цена: {price} руб.

                    Клиент задал вопрос: "{question}".
                    Ответьте профессионально и дружелюбно, используя данные о товаре.

                    Вопрос клиента: {question}

                    Ответ:
                    """
                )
                response = llm.invoke(product_context_template.format(
                    product_name=product_info["product_name"],
                    availability=product_info["availability"],
                    price=product_info["price"],
                    question=last_user_message
                ))
                response_text = response.content if isinstance(response, AIMessage) else str(response)
                logger.info("Сгенерирован ответ через модель с учетом информации о товаре.")
            else:
                llm = ChatOpenAI(api_key=openai_api_key)
                general_template = PromptTemplate.from_template(
                    """
                    Клиент задал вопрос: "{question}".
                    Ответьте максимально профессионально и дружелюбно.

                    Вопрос клиента: {question}

                    Ответ:
                    """
                )
                response = llm.invoke(general_template.format(question=last_user_message))
                response_text = response.content if isinstance(response, AIMessage) else str(response)
                logger.info("Сгенерирован общий ответ через модель.")

        # Сохраняем ответ в память и обновляем состояние
        user_memory.chat_memory.add_ai_message(response_text)
        logger.info(f"Ответ сохранен в память: {response_text}")
        state["messages"].append(HumanMessage(content=response_text))
        logger.info(f"Ответ клиенту: {response_text}")
        return state

    except Exception as e:
        error_message = "Произошла ошибка. Повторите запрос."
        logger.error(f"Ошибка при обработке узла графа: {e}", exc_info=True)
        user_memory = get_user_memory(state.get("user_id", "default_user"))
        user_memory.chat_memory.add_ai_message(error_message)
        state["messages"].append(HumanMessage(content=error_message))
        return state

# Создание графа состояний
graph_builder = StateGraph(state_schema=State)

# Добавление узлов в граф
def start_node(state: State) -> State:
    """
    Начальный узел для инициализации состояния.
    """
    return {"messages": [], "current_product": None}

graph_builder.add_node("start_node", start_node)

# Добавление узла "chatbot"
logger.info("Добавление узла 'chatbot' в граф состояний.")
graph_builder.add_node("chatbot", chatbot)

# Добавление связей в граф
logger.info("Добавление связи START -> start_node в граф состояний.")
graph_builder.add_edge(START, "start_node")
logger.info("Добавление связи start_node -> chatbot в граф состояний.")
graph_builder.add_edge("start_node", "chatbot")

logger.info("Добавление связи chatbot -> END в граф состояний.")
graph_builder.add_edge("chatbot", END)

# Компиляция графа состояний
logger.info("Компиляция графа состояний.")
graph = graph_builder.compile()

user_states = {}


# customer_chat_router = APIRouter()
# templates = Jinja2Templates(directory="templates")

@customer_chat_router.get('/customer/customer-chat', response_class=HTMLResponse)
async def render_customer_chat(request: Request):
    logger.info("Отображение страницы customer_chat.html.")
    return templates.TemplateResponse('customer_chat.html', {"request": request})

@customer_chat_router.post('/customer/customer-send-message')
async def respond_to_message(data: dict, db: Session = Depends(get_db)):
    try:
        logger.info(f"Полученные данные: {data}")

        user_message = data.get("question")
        user_id = data.get("user_id", "user124")

        if not user_message:
            logger.error("Ошибка: Вопрос не был передан в запросе.")
            raise HTTPException(status_code=400, detail="Пожалуйста, введите ваш вопрос.")

        # Получение памяти пользователя
        user_memory = get_user_memory(user_id)

        # Сохранение сообщения в память
        user_memory.chat_memory.add_user_message(user_message)

        # Извлекаем текущее состояние пользователя
        current_state = user_states.get(user_id, {"messages": [], "current_product": None})
        current_state["messages"].append(HumanMessage(content=user_message))

        # Запускаем граф состояний
        updated_state = await chatbot(current_state, SessionLocal)

        # Сохраняем обновлённое состояние
        user_states[user_id] = updated_state

        response_text = updated_state["messages"][-1].content

        # Сохраняем ответ в память
        user_memory.chat_memory.add_ai_message(response_text)

        # Сохраняем взаимодействие
        save_interaction(SessionLocal, user_id, user_message, response_text)

        logger.info(f"Ответ клиенту: {response_text}")
        return JSONResponse(content={"answer": [response_text]}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        return JSONResponse(content={"answer": ["Произошла ошибка. Повторите запрос."]}, status_code=500)
