# -*- coding: windows-1251 -*-

import re
import os
import json
import logging
from typing import Optional, TypedDict, Annotated
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_experimental.agents import create_csv_agent
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, START, END
from customer.classification import is_product_related
from customer.faiss import find_product_in_faiss

from customer.config import logger, openai_api_key
from db import get_db, SessionLocal
from models import PredefinedQuestion, Interaction

# Настройка шаблонов
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Инициализация переменных
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API ключ OpenAI не найден в переменных окружения")

# Инициализация LLM и эмбеддингов
llm = ChatOpenAI(api_key=openai_api_key, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Определение памяти пользователей
user_memories = {}
user_states = {}

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)
    return user_memories[user_id]

# Определяем тип состояния
class State(TypedDict):
    messages: list[dict]  # История сообщений в виде списка словарей
    current_product: Optional[dict]  # Информация о текущем товаре

# Чистка текста от HTML-тегов
def clean_html(text: str) -> str:
    return re.sub(r'<.*?>', '', text)

# Сохранение взаимодействия
def save_interaction(session: Session, user_id: str, query: str, response: str) -> None:
    try:
        interaction = Interaction(user_id=user_id, query=query, response=clean_html(response))
        session.add(interaction)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка сохранения взаимодействия: {e}", exc_info=True)

# Извлечение данных из сообщения
def extract_data(message: str) -> dict:
    prompt = PromptTemplate.from_template("""
    Извлеки данные в формате JSON:
    - product_name: название товара.
    - quantity: количество.
    - method: доставка/самовывоз.
    Сообщение: {message}
    """)
    response = llm.invoke(prompt.format(message=message))
    try:
        return json.loads(response.content if isinstance(response, AIMessage) else response)
    except json.JSONDecodeError:
        raise ValueError("Ошибка декодирования JSON из ответа.")

# Узел графа: обработка сообщений
async def chatbot(state: State, session: Session = Depends(get_db)) -> State:
    try:
        last_message = state["messages"][-1].content
        logger.info(f"Обработка сообщения: {last_message}")

        # Проверка, связан ли вопрос с товаром
        if is_product_related(last_message):
            product_info = find_product_in_faiss(last_message) or {}
            state["current_product"] = product_info

            if product_info:
                response = f"{product_info.get('product_name', 'Товар')} в наличии: {product_info.get('availability', 'неизвестно')} шт. Цена: {product_info.get('price', 'неизвестно')} руб."
            else:
                response = "Не удалось найти информацию о товаре."
        else:
            prompt = PromptTemplate.from_template("""
            Клиент задал вопрос: "{question}". Ответьте профессионально и дружелюбно.
            Вопрос клиента: {question}
            Ответ:
            """)
            response = llm.invoke(prompt.format(question=last_message)).content

        state["messages"].append(HumanMessage(content=response))
        logger.info(f"Сгенерированный ответ: {response}")
        return state
    except Exception as e:
        logger.error(f"Ошибка в узле графа: {e}", exc_info=True)
        state["messages"].append(HumanMessage(content="Произошла ошибка, повторите запрос."))
        return state

# Инициализация графа
graph_builder = StateGraph(state_schema=State)
graph_builder.add_node("start_node", lambda state: {"messages": [], "current_product": None})
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "start_node")
graph_builder.add_edge("start_node", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Роутеры
customer_chat_router = APIRouter()

@customer_chat_router.get("/customer/customer-chat", response_class=HTMLResponse)
async def render_customer_chat(request: Request):
    return templates.TemplateResponse("customer_chat.html", {"request": request})

@customer_chat_router.post("/customer/customer-send-message")
async def respond_to_message(data: dict, db: Session = Depends(get_db)):
    try:
        user_message = data.get("question")
        user_id = data.get("user_id", "user124")

        if not user_message:
            raise HTTPException(status_code=400, detail="Вопрос не предоставлен.")

        # Извлечение состояния пользователя
        user_state = user_states.get(user_id, {"messages": [], "current_product": None})
        user_state["messages"].append(HumanMessage(content=user_message))

        # Запуск графа
        updated_state = await chatbot(user_state, db)

        # Сохранение взаимодействия
        response_text = updated_state["messages"][-1].content
        save_interaction(db, user_id, user_message, response_text)

        # Обновление состояния пользователя
        user_states[user_id] = updated_state
        return JSONResponse(content={"answer": [response_text]})
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}", exc_info=True)
        return JSONResponse(content={"answer": ["Произошла ошибка. Повторите запрос."]}, status_code=500)
