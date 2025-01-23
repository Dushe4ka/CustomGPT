# -*- coding: utf-8 -*-

import re
import json
import logging
from typing import List, Optional, TypedDict
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, START, END
from customer.classification import is_product_related
from customer.faiss import find_product_in_faiss
from customer.get_template import find_question_template
from customer.config import logger, openai_api_key
from db import get_db, SessionLocal
from models import Interaction

import os


# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Инициализация моделей OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API ключ OpenAI не найден в переменных окружения")

llm = ChatOpenAI(openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Инициализация памяти для пользователей
user_memories = {}
user_states = {}

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)
    return user_memories[user_id]

# Определение типа состояния
class State(TypedDict):
    messages: List[HumanMessage]  # История сообщений
    current_product: Optional[dict]  # Информация о текущем обсуждаемом товаре

def clean_html(text: str) -> str:
    return re.sub(r'<.*?>', '', text)

def save_interaction(session: Session, user_id: str, query: str, response: str) -> None:
    try:
        interaction = Interaction(user_id=user_id, query=query, response=clean_html(response))
        session.add(interaction)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка сохранения взаимодействия: {e}", exc_info=True)

def chatbot(state: State, session: Session) -> State:
    try:
        last_message = state["messages"][-1].content
        logger.info(f"Обработка сообщения: {last_message}")

        if is_product_related(last_message):
            product_info = find_product_in_faiss(last_message) or {}
            state["current_product"] = product_info

            if product_info:
                template = find_question_template(last_message)
                response = template.format(
                    product_name=product_info.get("product_name", "неизвестно"),
                    availability=product_info.get("availability", "неизвестно"),
                    price=product_info.get("price", "неизвестно")
                )
            else:
                response = "Не удалось найти информацию о товаре."
        else:
            prompt = PromptTemplate.from_template(
                """
                Клиент задал вопрос: "{question}". Ответьте профессионально и дружелюбно.
                Вопрос клиента: {question}
                Ответ:
                """
            )
            response = llm.invoke(prompt.format(question=last_message)).content

        state["messages"].append(HumanMessage(content=response))
        logger.info(f"Сгенерированный ответ: {response}")
        return state
    except Exception as e:
        logger.error(f"Ошибка в узле графа: {e}", exc_info=True)
        state["messages"].append(HumanMessage(content="Произошла ошибка, повторите запрос."))
        return state

# Создание графа состояний
graph_builder = StateGraph(state_schema=State)

def start_node(state: State) -> State:
    return {"messages": [], "current_product": None}

graph_builder.add_node("start_node", start_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "start_node")
graph_builder.add_edge("start_node", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Создание API роутера
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

        user_state = user_states.get(user_id, {"messages": [], "current_product": None})
        user_state["messages"].append(HumanMessage(content=user_message))

        updated_state = chatbot(user_state, db)

        response_text = updated_state["messages"][-1].content
        save_interaction(db, user_id, user_message, response_text)

        user_states[user_id] = updated_state
        return JSONResponse(content={"answer": [response_text]})
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}", exc_info=True)
        return JSONResponse(content={"answer": ["Произошла ошибка. Повторите запрос."]}, status_code=500)
