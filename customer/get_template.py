# -*- coding: windows-1251 -*-
import os
import faiss
import numpy as np
from sqlalchemy.sql import text
from sqlalchemy.orm import Session
from langchain_openai import OpenAIEmbeddings
from typing import Optional
from customer.config import SessionLocal, logger, openai_api_key
from customer.faiss import save_question_index
from models import PredefinedQuestion

# ��������� �������� ��� ������ � FAISS
FAISS_THRESHOLD = 0.5


def get_response_template(session: Session, category: str, stage: str) -> Optional[str]:
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
            logger.warning("������ �������� �����������. ������ ����� ������.")
            save_question_index()

        # ������ ������� � ��������������� ��������
        index = faiss.read_index("question_index.faiss")
        question_ids = np.load("question_ids.npy")
        logger.info("������ �������� � �������������� ������� ���������.")

        # ��������� ���������� ��� �������
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)

        # ����� ��������� �������
        D, I = index.search(np.array([query_vector]), k=1)
        logger.info(f"���������� ������: D={D}, I={I}")

        # �������� ����������
        if len(I[0]) > 0 and D[0][0] < FAISS_THRESHOLD:
            matched_question_id = question_ids[I[0][0]]
            logger.info(f"������ ���������� ������ � ID: {matched_question_id}")

            # ��������� ������ ������ �� ���� ������
            with SessionLocal() as session:
                query = text("SELECT answer FROM predefined_questions WHERE id = :id")
                result = session.execute(query, {"id": int(matched_question_id)}).scalar()

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
