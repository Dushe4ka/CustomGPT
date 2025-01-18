# -*- coding: windows-1251 -*-
from models import PredefinedQuestion  # ������ �� ���� ������
from db import SessionLocal
import os  # ��� �������� ������������� ������
import faiss  # ��� ������ � FAISS ���������
import numpy as np  # ��� ������ � ��������� ���������������
from langchain_openai import OpenAIEmbeddings  # ��� ��������� �����������
from customer.config import SessionLocal, logger, openai_api_key  # ��� ������ � ����� ������, ����������� � ���� API
from sqlalchemy.sql import text  # ��� ���������� SQL-��������
from typing import Optional  # ��� ��������� ������������� ����
from customer.faiss import save_question_index  # ��� ������������ FAISS �������

def get_response_template(session: SessionLocal, category: str, stage: str) -> Optional[str]:
    """
    �������� ������ ������ �� ������� predefined_questions �� ������ ��������� � ������ �������.

    :param session: SQLAlchemy ������ ��� ���������� ��������.
    :param category: ��������� �������.
    :param stage: ������ �������.
    :return: ������ ������ ��� None, ���� ������ �� ������.
    """
    try:
        logger.info(f"������ ������� ��� ��������� '{category}' � ������ '{stage}'.")

        # ��������� ������ � ������� predefined_questions
        template = (
            session.query(PredefinedQuestion.answer)
            .filter_by(category=category, stage=stage)
            .scalar()
        )

        if template:
            logger.info(f"������ ������ ��� ��������� '{category}' � ������ '{stage}'.")
        else:
            logger.warning(f"������ ��� ��������� '{category}' � ������ '{stage}' �� ������.")

        return template
    except Exception as e:
        logger.error(f"������ ��� ��������� ������� ��� ��������� '{category}' � ������ '{stage}': {e}")
        return None

def find_question_template(query: str) -> Optional[str]:
    """
    ������� ������ ������ �� ������ ������� ������������, ��������� FAISS.
    """
    logger.info(f"���� ���������� ������ ��� �������: '{query}'")
    try:
        # ��������� ������������� ������
        if not os.path.exists("question_index.faiss") or not os.path.exists("question_ids.npy"):
            logger.info("������ �������� �����������. ������ ����� ������.")
            save_question_index()

        # ������ ������� � ��������������� ��������
        index = faiss.read_index("question_index.faiss")
        question_ids = np.load("question_ids.npy")
        logger.info("������ �������� � �������������� ������� ���������.")

        # ��������� ���������� ��� �������
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)

        # ����� ��������� �������
        D, I = index.search(np.array([query_vector]), k=1)
        logger.info(f"���������� ������: D={D}, I={I}")

        # �������� ����������
        if len(I[0]) > 0 and D[0][0] < 0.5:  # ��������� �������� ����� ���������
            matched_question_id = question_ids[I[0][0]]
            logger.info(f"������ ���������� ������ � ID: {matched_question_id}")

            # ��������� ������ ������ �� ���� ������
            session = SessionLocal()
            query = text("SELECT answer FROM predefined_questions WHERE id = :id")
            result = session.execute(query, {"id": int(matched_question_id)}).scalar()
            session.close()

            if result:
                logger.info(f"������ ������: {result}")
                return result
            else:
                logger.warning("������ ������ �� ������ � ���� ������.")
        else:
            logger.warning("���������� ������ �� ������ � �������.")
        return None
    except Exception as e:
        logger.error(f"������ ��� ������ ������� ����� FAISS: {e}", exc_info=True)
        return None