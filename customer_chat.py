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

# ��������� ��������
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

# ������������� ����������
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API ���� OpenAI �� ������ � ���������� ���������")

# ������������� LLM � �����������
llm = ChatOpenAI(api_key=openai_api_key, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# ����������� ������ �������������
user_memories = {}
user_states = {}

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)
    return user_memories[user_id]

# ���������� ��� ���������
class State(TypedDict):
    messages: list[dict]  # ������� ��������� � ���� ������ ��������
    current_product: Optional[dict]  # ���������� � ������� ������

# ������ ������ �� HTML-�����
def clean_html(text: str) -> str:
    return re.sub(r'<.*?>', '', text)

# ���������� ��������������
def save_interaction(session: Session, user_id: str, query: str, response: str) -> None:
    try:
        interaction = Interaction(user_id=user_id, query=query, response=clean_html(response))
        session.add(interaction)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"������ ���������� ��������������: {e}", exc_info=True)

# ���������� ������ �� ���������
def extract_data(message: str) -> dict:
    prompt = PromptTemplate.from_template("""
    ������� ������ � ������� JSON:
    - product_name: �������� ������.
    - quantity: ����������.
    - method: ��������/���������.
    ���������: {message}
    """)
    response = llm.invoke(prompt.format(message=message))
    try:
        return json.loads(response.content if isinstance(response, AIMessage) else response)
    except json.JSONDecodeError:
        raise ValueError("������ ������������� JSON �� ������.")

# ���� �����: ��������� ���������
async def chatbot(state: State, session: Session = Depends(get_db)) -> State:
    try:
        last_message = state["messages"][-1].content
        logger.info(f"��������� ���������: {last_message}")

        # ��������, ������ �� ������ � �������
        if is_product_related(last_message):
            product_info = find_product_in_faiss(last_message) or {}
            state["current_product"] = product_info

            if product_info:
                response = f"{product_info.get('product_name', '�����')} � �������: {product_info.get('availability', '����������')} ��. ����: {product_info.get('price', '����������')} ���."
            else:
                response = "�� ������� ����� ���������� � ������."
        else:
            prompt = PromptTemplate.from_template("""
            ������ ����� ������: "{question}". �������� ��������������� � ����������.
            ������ �������: {question}
            �����:
            """)
            response = llm.invoke(prompt.format(question=last_message)).content

        state["messages"].append(HumanMessage(content=response))
        logger.info(f"��������������� �����: {response}")
        return state
    except Exception as e:
        logger.error(f"������ � ���� �����: {e}", exc_info=True)
        state["messages"].append(HumanMessage(content="��������� ������, ��������� ������."))
        return state

# ������������� �����
graph_builder = StateGraph(state_schema=State)
graph_builder.add_node("start_node", lambda state: {"messages": [], "current_product": None})
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "start_node")
graph_builder.add_edge("start_node", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# �������
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
            raise HTTPException(status_code=400, detail="������ �� ������������.")

        # ���������� ��������� ������������
        user_state = user_states.get(user_id, {"messages": [], "current_product": None})
        user_state["messages"].append(HumanMessage(content=user_message))

        # ������ �����
        updated_state = await chatbot(user_state, db)

        # ���������� ��������������
        response_text = updated_state["messages"][-1].content
        save_interaction(db, user_id, user_message, response_text)

        # ���������� ��������� ������������
        user_states[user_id] = updated_state
        return JSONResponse(content={"answer": [response_text]})
    except Exception as e:
        logger.error(f"������ ��������� ���������: {e}", exc_info=True)
        return JSONResponse(content={"answer": ["��������� ������. ��������� ������."]}, status_code=500)
