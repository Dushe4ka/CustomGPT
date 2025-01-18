# -*- coding: windows-1251 -*-
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from sqlalchemy.sql import text
import os  # �������� ���� ������


import faiss
from customer.config import SessionLocal, openai_api_key, logger
load_dotenv()

# �������� ����������� �� ���� ������
def create_embeddings_from_db():
    session = SessionLocal()
    try:
        products = session.execute(text("SELECT id, name FROM base_product")).fetchall()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        product_ids = []
        product_vectors = []

        for product_id, product_name in products:
            vector = embeddings.embed_query(product_name)
            product_ids.append(product_id)
            product_vectors.append(vector)

        return product_ids, np.array(product_vectors)
    finally:
        session.close()

# �������� FAISS �������
def save_faiss_index():
    product_ids, product_vectors = create_embeddings_from_db()
    dimension = len(product_vectors[0])  # ����������� ����������
    index = faiss.IndexFlatL2(dimension)
    index.add(product_vectors)

    # ��������� ������ � ������������ product_ids
    faiss.write_index(index, "product_index.faiss")
    np.save("product_ids.npy", np.array(product_ids))

# ����� ������ � FAISS
def find_product_with_faiss(query: str):
    logger.info(f"���� ����� �� �������: '{query}'")
    try:
        # ��������� ������������� ������
        if not os.path.exists("product_index.faiss") or not os.path.exists("product_ids.npy"):
            logger.info("FAISS ������ �����������. ������ ����� ������.")
            save_faiss_index()

        # ������ ������� � ��������������� �������
        index = faiss.read_index("product_index.faiss")
        product_ids = np.load("product_ids.npy")
        logger.info("FAISS ������ � �������������� ������� ���������.")

        # ��������� ���������� ��� �������
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logger.info("��������� ��� ������� ������� ������.")

        # ����� ��������� �������
        D, I = index.search(np.array([query_vector]), k=1)
        logger.info(f"���������� ������: D={D}, I={I}")

        # �������� ����������
        if len(I[0]) > 0 and D[0][0] < 0.5:
            matched_product_id = product_ids[I[0][0]]
            logger.info(f"������ ����� � ID: {matched_product_id}, ����������: {D[0][0]}")
            return matched_product_id
        else:
            logger.warning("����� �� ������ � FAISS.")
            return None
    except Exception as e:
        logger.error(f"������ ��� ������ ������ � FAISS: {e}", exc_info=True)
        return None

def get_product_by_id(product_id: int):
    """
    ��������� ���������� � ������ �� ���� ������ �� ��� ID.

    Args:
        product_id (int): ���������� ������������� ������.

    Returns:
        dict: ���������� � ������ � ������� 'product_name', 'availability' � 'price'.
        None: ���� ����� �� ������ ��� ��������� ������.
    """
    logger.info(f"��������� ���������� � ������ � ID: {product_id}")
    session = SessionLocal()
    try:
        # ��������, ��� product_id ����� ���������� ���
        product_id = int(product_id)

        # ��������� ������ � ���� ������
        query = text("SELECT name, quantity, price FROM base_product WHERE id = :id")
        product = session.execute(query, {"id": product_id}).fetchone()

        # ��������� ���������
        if product:
            product_data = {
                "product_name": product[0],
                "availability": product[1],
                "price": product[2]
            }
            logger.info(f"���������� � ������: {product_data}")
            return product_data
        else:
            logger.warning(f"����� � ID {product_id} �� ������ � ���� ������.")
            return None
    except ValueError as ve:
        logger.error(f"�������� ��� ������ ��� product_id: {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"������ ��� ��������� ���������� � ������: {e}", exc_info=True)
        return None
    finally:
        # ��������� ������ � ����� ������
        session.close()

def find_product_in_faiss(query: str) -> dict:
    logger.info(f"���� ����� � FAISS �� �������: '{query}'")
    try:
        # ����� ������ � FAISS
        product_id = find_product_with_faiss(query)
        if product_id:
            # ��������� ���������� � ������ �� ���� ������
            product = get_product_by_id(product_id)
            if product:
                logger.info(f"����� ������� ������: {product}")
                return product
        logger.warning("����� �� ������ ����� FAISS.")
        return None
    except Exception as e:
        logger.error(f"������ ��� ������ ������ ����� FAISS: {e}", exc_info=True)
        return None

def create_question_embeddings():
    """
    ������ ���������� ��� �������� �� ������� predefined_questions.
    """
    session = SessionLocal()
    try:
        # ��������� ������� � �� ID �� ������� predefined_questions
        questions = session.execute(text("SELECT id, question FROM predefined_questions")).fetchall()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        question_ids = []
        question_vectors = []

        for question_id, question_text in questions:
            if question_text:  # ���������, ��� ����� ������� �� ������
                vector = embeddings.embed_query(question_text)
                question_ids.append(question_id)
                question_vectors.append(vector)

        return question_ids, np.array(question_vectors)
    finally:
        session.close()


def save_question_index():
    """
    ������ � ��������� FAISS ������ ��� ��������.
    """
    question_ids, question_vectors = create_question_embeddings()
    dimension = len(question_vectors[0])  # ����������� �����������
    index = faiss.IndexFlatL2(dimension)
    index.add(question_vectors)

    # ��������� ������ � ������������ question_ids
    faiss.write_index(index, "question_index.faiss")
    np.save("question_ids.npy", np.array(question_ids))