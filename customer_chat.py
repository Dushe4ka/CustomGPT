# -*- coding: windows-1251 -*-

import re
from langchain.schema import AIMessage
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAIEmbeddings

from sqlalchemy.sql import text
from starlette.responses import HTMLResponse

from customer.config import openai_api_key
from db import SessionLocal, get_db
from models import PredefinedQuestion  # ����������� ������ ��������������
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

# ��������� ��������
templates = Jinja2Templates(directory="templates")

# �������� ������� ��� ���� � ���������
customer_chat_router = APIRouter()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# ������ ����������� ����������
import pandas as pd
import json
import logging
from dotenv import load_dotenv
import spacy

# Pydantic ��� ��������� ������
from pydantic import BaseModel, Field
from typing import Optional

import os
from langchain.schema import HumanMessage

# ������� ��� ������� HTML-����� �� ������
def clean_html(text):
    logger.debug(f"������� ����� �� HTML-�����: {text[:100]}...")  # �������� ������ ������
    clean_text = re.sub(r'<.*?>', '', text)  # ������� ��� HTML ����
    logger.debug(f"��������� �����: {clean_text[:100]}...")  # �������� ������ ���������� ������
    return clean_text

# �������� ���������� ���������
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API ���� OpenAI �� ������ � ���������� ���������")

# �������� ������ ��� �������� �����
nlp = spacy.load('ru_core_news_sm')

# ������������� ������� OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)
llm2 = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# ������ ��� �������� ���������� � ������
class OrderDetails(BaseModel):
    """���������� � ������, ����������� �� ������� ������������."""
    quantity: Optional[int] = Field(default=None, description="���������� �������, ������� ������������ ����� ��������")
    method: Optional[str] = Field(default=None, description="����� ��������� ������: �������� ��� ���������")


# �������� ������������� CSV ����� � ��� ��������, ���� �� �����������
csv_file = "product_converted.csv"
if not os.path.exists(csv_file):
    df = pd.read_excel("product.xls")
    df.to_csv(csv_file, index=False)

# �������� ������ ��� ������ � CSV ������
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)

# templates = Jinja2Templates(directory="static")

@customer_chat_router.get("/customer_chat")
async def customer_chat(request: Request):
    return templates.TemplateResponse("customer_chat.html", {"request": request})

# ������ �������� ������� ��� ���������� ������
extraction_chain = (
    PromptTemplate(input_variables=["message"], template="""������� ������ � ������� JSON:
    - product_name: �������� ������.
    - quantity: ����������.
    - method: ��������/���������.
    ���������: {message}""")
    | ChatOpenAI(api_key=openai_api_key)  # ����������� ���� ������ OpenAI
)

def extract_data(message):
    """
    ��������� ������ �� ��������� ������������.

    Args:
        message (str): ��������� ������������.

    Returns:
        dict: ����������� ������ � ������� JSON.

    Raises:
        ValueError: ���� ��������� ����� ����������� ������.
        Exception: ��� ����� ������ ������.
    """
    try:
        logger.info(f"������ ���������� ���������� �� ���������: {message}")

        try:
            # ��� 1: ������ ������� ����������
            extraction_result = extraction_chain.invoke({"message": message})

            # ����������� ����������
            logger.info(f"����� ��������� ����������: {extraction_result}")
            logger.info(f"��� ���������� ����������: {type(extraction_result)}")
        except Exception as e:
            logger.error(f"������ ��� ������ ������� ����������: {str(e)}")
            raise ValueError("������ ��� ������ ������� ���������� ������.")

        try:
            # ��� 2: ��������� ����������
            extraction_text = None
            if isinstance(extraction_result, AIMessage):
                extraction_text = extraction_result.content
                logger.debug(f"��������� �������� �� AIMessage: {extraction_text}")
            elif isinstance(extraction_result, list) and len(extraction_result) > 0:
                extraction_text = extraction_result[0].content
                logger.debug(f"��������� �������� �� ������: {extraction_text}")
            elif isinstance(extraction_result, str):
                extraction_text = extraction_result
                logger.debug(f"��������� �������� �� ������: {extraction_text}")
            else:
                logger.error(f"����������� ������ ���������� ����������: {type(extraction_result)}")
                raise ValueError(f"����������� ������ ���������� ����������: {type(extraction_result)}")

            # ��������, ���� �� ���������
            if not extraction_text:
                logger.warning("��������� ���������� ������.")
                raise ValueError("��������� ���������� �� �������� ������.")
        except Exception as e:
            logger.error(f"������ ��������� ���������� ����������: {str(e)}")
            raise ValueError("������ ��������� ���������� ���������� ������.")

        try:
            # ��� 3: �������������� � JSON
            extracted_data = json.loads(extraction_text.replace("'", '"'))
            logger.info(f"����������� ������ � ������� JSON: {extracted_data}")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"������ ������������� JSON: {e}")
            raise ValueError("�� ������� ������������� ��������� ���������� � JSON.")

    except Exception as e:
        logger.error(f"������ ��� ���������� ������: {str(e)}")
        raise e

def save_interaction(session: Session, user_id: str, query: str, response: str) -> None:
    """
    ��������� �������������� � ���� ������ � �������������� SQLAlchemy.

    :param session: SQLAlchemy ������ ��� ���������� ��������.
    :param user_id: ������������� ������������.
    :param query: ������ ������������.
    :param response: �����, ������� ����� �������.
    """
    try:
        logger.debug(f"��������� �������������� ��� user_id={user_id} � ��������: {query}")

        # ������� ������ �� HTML-�����
        response_cleaned = clean_html(response)

        # �������� ��������� ����� ����� �����������
        logger.debug(f"��������� �����: {response_cleaned[:100]}...")  # �������� ������ ���������� ������

        # ������ ������ ��������������
        interaction = Interaction(user_id=user_id, query=query, response=response_cleaned)

        # ��������� � ��������� � ���� ������
        session.add(interaction)
        session.commit()

        logger.info(f"�������������� ������� ��������� ��� user_id={user_id}")

    except Exception as e:
        session.rollback()  # ���������� ��������� � ������ ������
        logger.error(f"������ ������ ��������������: {e}")


def get_message_history(user_id):
    """
    ��������� ������� ��������� ������������ �� ���� ������.
    """
    try:
        # ��������� ��� �������������� ��� ������������, �������� �� �������
        interactions = Interaction.query.filter(Interaction.user_id == user_id).order_by(Interaction.timestamp).all()

        # ��������� ������� ���������
        history = []
        for interaction in interactions:
            history.append({
                "user_message": interaction.query,  # ������ ������������
                "response": interaction.response  # �����
            })

        # �������� ���������� ����������� ��������������
        logger.info(f"��������� {len(history)} �������������� ��� user_id={user_id}")

        # �������� ������ �������������� �� �����������
        for idx, interaction in enumerate(history):
            logger.info(f"�������������� #{idx + 1}: ������ - {interaction['user_message']}, ����� - {interaction['response']}")

        return history

    except Exception as e:
        # �������� ������, ���� ���-�� ����� �� ���
        logger.error(f"������ ���������� ������� ���������: {e}")
        return []


# ���������� ��� ���������
class State(TypedDict):
    messages: Annotated[list, add_messages]  # ������� ���������
    current_product: dict  # ���������� � ������� ����������� ������


def get_user_state(user_id):
    """���������� ��������� ������������ ��� ������ �����, ���� ��� ���."""
    if user_id not in user_states:
        user_states[user_id] = {"messages": [], "current_product": None}
    return user_states[user_id]

def classify_message_with_openai(message: str) -> bool:
    """
    ����������, ������ �� ������ � �������, ��������� OpenAI.
    """
    prompt = f"""
    ������: "{message}"
    �������� "��", ���� ������ ������ � ������� (��������, ��������, �����, ����������������), 
    ��� "���" � ��������� ������.
    """
    response = llm.invoke(prompt)
    return "��" in response.content.lower()


#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ������������� ������ ��� ������� ������������
user_memories = {}

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)  # ������� ��������� 10 ���������
    return user_memories[user_id]


# �������-���� �����
async def chatbot(state: State, session: Session = Depends(get_db)) -> State:
    """
    �������-���� ����� ��� ��������� ��������� ������� � ������ ������ � ���������.
    """
    try:
        logger.info(f"������ ���������. ������� ���������: {state}")

        # ���������� ���������� ��������� ������������
        messages = state.get("messages", [])
        last_user_message = messages[-1].content if messages else "������, ���!"
        logger.info(f"��������� ��������� ������������: {last_user_message}")

        # ��������� ������ ������������
        user_memory = get_user_memory(state.get("user_id", "default_user"))

        # ��������� ��������� ������������ � ������
        user_memory.chat_memory.add_user_message(last_user_message)
        logger.info("��������� ������������ ��������� � ������.")

        # ���������, ������ �� ������ � �������
        logger.info(f"���������, ������ �� ������ � �������: '{last_user_message}'")
        is_related = is_product_related(last_user_message)
        logger.info(f"��������� �������������: {'��' if is_related else '���'} ��� �������: '{last_user_message}'")

        if is_related:
            logger.info("������ ������ � �������.")
            # ������� ����� ������ ������ ����� FAISS
            template = find_question_template(last_user_message)
            current_product = state.get("current_product")

            if template:
                logger.info(f"������ ������ ������ ����� FAISS: {template}")

                # ���� ���������� � ������ ���, ���� ����� ����� FAISS
                if not current_product:
                    logger.info("������� ���������� � ������ �����������. �������� ����� ����� ����� FAISS.")
                    current_product = find_product_in_faiss(last_user_message)
                    if current_product:
                        state["current_product"] = current_product
                        logger.info(f"���������� � ��������� ������ ��������� � ���������: {current_product}")

                # ��������� �����, ���������� ������ � ������ � ������
                if current_product and all(key in current_product for key in ["product_name", "availability", "price"]):
                    response_text = template.format(
                        product_name=current_product["product_name"],
                        availability=current_product["availability"],
                        price=current_product["price"]
                    )
                    # ��������� ���������� � ������ � ������
                    memory_message = (
                        f"��������� �����: {current_product['product_name']}. "
                        f"��������: {current_product['availability']} ��. ����: {current_product['price']} ���."
                    )
                    user_memory.chat_memory.add_ai_message(memory_message)
                    logger.info(f"���������� � ������ ��������� � ������: {memory_message}")
                else:
                    logger.warning("���������� � ������ ����������� ��� �����������. ���������� ��������� ��������.")
                    response_text = template.format(
                        product_name="����������� �����",
                        availability="����������",
                        price="����������"
                    )
            else:
                logger.warning("������ �� ������ ����� FAISS. ����������� ����� ������.")
                product_info = find_product_in_faiss(last_user_message)

                if product_info:
                    logger.info(f"������ �����: {product_info}")
                    state["current_product"] = product_info
                    response_text = (
                        f"{product_info['product_name']}' � �������: "
                        f"{product_info['availability']} ��. �����: {product_info['price']} ���. "
                        f"������� ��� �����?"
                    )
                    memory_message = (
                        f"��������� �����: {product_info['product_name']}. "
                        f"��������: {product_info['availability']} ��. ����: {product_info['price']} ���."
                    )
                    user_memory.chat_memory.add_ai_message(memory_message)
                    logger.info(f"���������� � ��������� ������ ��������� � ������: {memory_message}")
                else:
                    logger.warning("����� �� ������.")
                    response_text = "���� ������, ������"
        else:
            logger.info("������ �� ������ � �������. ��������� ������ ������ ����� ���������.")
            # ���������, ���� �� ����������� ������ � ������
            if state.get("current_product"):
                product_info = state["current_product"]
                logger.info("������������ ���������� � ������ �� ���������.")

                # �������� ���������� � ������ � ������ ��� ������������ ������
                llm = ChatOpenAI(api_key=openai_api_key)
                product_context_template = PromptTemplate.from_template(
                    """
                    �������� ��������� ���������� � ������:
                    ��������: {product_name}
                    � �������: {availability} ��.
                    ����: {price} ���.

                    ������ ����� ������: "{question}".
                    �������� ��������������� � ����������, ��������� ������ � ������.

                    ������ �������: {question}

                    �����:
                    """
                )
                response = llm.invoke(product_context_template.format(
                    product_name=product_info["product_name"],
                    availability=product_info["availability"],
                    price=product_info["price"],
                    question=last_user_message
                ))
                response_text = response.content if isinstance(response, AIMessage) else str(response)
                logger.info("������������ ����� ����� ������ � ������ ���������� � ������.")
            else:
                llm = ChatOpenAI(api_key=openai_api_key)
                general_template = PromptTemplate.from_template(
                    """
                    ������ ����� ������: "{question}".
                    �������� ����������� ��������������� � ����������.

                    ������ �������: {question}

                    �����:
                    """
                )
                response = llm.invoke(general_template.format(question=last_user_message))
                response_text = response.content if isinstance(response, AIMessage) else str(response)
                logger.info("������������ ����� ����� ����� ������.")

        # ��������� ����� � ������ � ��������� ���������
        user_memory.chat_memory.add_ai_message(response_text)
        logger.info(f"����� �������� � ������: {response_text}")
        state["messages"].append(HumanMessage(content=response_text))
        logger.info(f"����� �������: {response_text}")
        return state

    except Exception as e:
        error_message = "��������� ������. ��������� ������."
        logger.error(f"������ ��� ��������� ���� �����: {e}", exc_info=True)
        user_memory = get_user_memory(state.get("user_id", "default_user"))
        user_memory.chat_memory.add_ai_message(error_message)
        state["messages"].append(HumanMessage(content=error_message))
        return state

# �������� ����� ���������
graph_builder = StateGraph(state_schema=State)

# ���������� ����� � ����
def start_node(state: State) -> State:
    """
    ��������� ���� ��� ������������� ���������.
    """
    return {"messages": [], "current_product": None}

graph_builder.add_node("start_node", start_node)

# ���������� ���� "chatbot"
logger.info("���������� ���� 'chatbot' � ���� ���������.")
graph_builder.add_node("chatbot", chatbot)

# ���������� ������ � ����
logger.info("���������� ����� START -> start_node � ���� ���������.")
graph_builder.add_edge(START, "start_node")
logger.info("���������� ����� start_node -> chatbot � ���� ���������.")
graph_builder.add_edge("start_node", "chatbot")

logger.info("���������� ����� chatbot -> END � ���� ���������.")
graph_builder.add_edge("chatbot", END)

# ���������� ����� ���������
logger.info("���������� ����� ���������.")
graph = graph_builder.compile()

user_states = {}


# customer_chat_router = APIRouter()
# templates = Jinja2Templates(directory="templates")

@customer_chat_router.get('/customer/customer-chat', response_class=HTMLResponse)
async def render_customer_chat(request: Request):
    logger.info("����������� �������� customer_chat.html.")
    return templates.TemplateResponse('customer_chat.html', {"request": request})

@customer_chat_router.post('/customer/customer-send-message')
async def respond_to_message(data: dict, db: Session = Depends(get_db)):
    try:
        logger.info(f"���������� ������: {data}")

        user_message = data.get("question")
        user_id = data.get("user_id", "user124")

        if not user_message:
            logger.error("������: ������ �� ��� ������� � �������.")
            raise HTTPException(status_code=400, detail="����������, ������� ��� ������.")

        # ��������� ������ ������������
        user_memory = get_user_memory(user_id)

        # ���������� ��������� � ������
        user_memory.chat_memory.add_user_message(user_message)

        # ��������� ������� ��������� ������������
        current_state = user_states.get(user_id, {"messages": [], "current_product": None})
        current_state["messages"].append(HumanMessage(content=user_message))

        # ��������� ���� ���������
        updated_state = await chatbot(current_state, SessionLocal)

        # ��������� ���������� ���������
        user_states[user_id] = updated_state

        response_text = updated_state["messages"][-1].content

        # ��������� ����� � ������
        user_memory.chat_memory.add_ai_message(response_text)

        # ��������� ��������������
        save_interaction(SessionLocal, user_id, user_message, response_text)

        logger.info(f"����� �������: {response_text}")
        return JSONResponse(content={"answer": [response_text]}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"������ ��������� �������: {e}", exc_info=True)
        return JSONResponse(content={"answer": ["��������� ������. ��������� ������."]}, status_code=500)
