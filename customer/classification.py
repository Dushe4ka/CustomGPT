# -*- coding: windows-1251 -*-

import logging
import os
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from customer.config import openai_api_key, logger


load_dotenv()



# ������ �������������
classifier_chain = (
    PromptTemplate.from_template(
        """�������� ��������� ������ ������������, ��������������� ��� ���:
        - `������� ������`,
        - `��������� ������`,
        - `������`.

        ������� ����� � ������������ � ��������� ���������.

        <������>
        {question}
        </������>

        �������������:"""
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)

extraction_template = PromptTemplate(
    input_variables=["message"],
    template="""������� ������ � ������� JSON:
    - product_name: �������� ������.
    - quantity: ����������.
    - method: ��������/���������.
    ���������: {message}"""
)

#������������� ������ �������
stage_classifier_chain = (
    PromptTemplate.from_template(
        """�������� ��������� ������������, ���������� ������ �������:
        - `��������� �������`
        - `��������� ���������`
        - `����� ���������`
        - `���������`.

        ��������� ������������: {user_message}
        ������ �������:"""
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)

# ������ ������� �������������
classification_chain = (
    PromptTemplate.from_template(
        """
        �������� ��������� ������ ������������, ����������, ��������� �� �� � ������:
        - ���� ������ ������ � ������� (��������, ������ � �������, ����, ���������������, �������� ������), �������� "��".
        - ���� ������ �� ������ � ������� (��������, ����� ������ � ��������, ������� ��� ��������), �������� "���".
        ������ ������������: {message}
        �����:
        """
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)


def is_product_related(message: str) -> bool:
    """
    ����������, ������ �� ������ ������������ � �������.

    Args:
        message (str): ��������� ������������.

    Returns:
        bool: True, ���� ������ ������ � �������, ����� False.
    """
    try:
        logger.info(f"������� ��������� � ������ ��� �������������: '{message}'")
        response = classification_chain.invoke({"message": message})
        logger.info(f"����� ������: '{response.strip()}' ��� ���������: '{message}'")

        # ������� ������ ������
        clean_response = response.strip().lower().replace(".", "").replace(",", "")
        logger.info(f"��������� ����� ������: '{clean_response}'")

        if clean_response not in ["��", "���"]:
            logger.warning(f"������������� ����� ������: '{clean_response}'. �������, ��� ������ �� ������ � �������.")
            return False
        return clean_response == "��"
    except Exception as e:
        logger.error(f"������ ��� ���������� �������������: {e}", exc_info=True)
        return False


#���������� ��������� ������������
def determine_stage(state: dict, user_message: str) -> str:
    """
    ���������� ������ ������� �� ������ ��������� ������������.

    :param state: ������� ��������� ������������.
    :param user_message: ��������� �� ������������.
    :return: ����������� ������ �������.
    """
    try:
        stage = stage_classifier_chain.invoke({"user_message": user_message}).strip()
        stage = stage.strip('`')  # ������� ������ �������� ���������
        state["current_stage"] = stage
        logger.info(f"��������� ������ �������: {stage}")
        return stage
    except Exception as e:
        logger.error(f"������ ��� ����������� ������: {e}")
        state["current_stage"] = "���������"
        return "���������"

# ������������� �������
def classify_question(question):
    classification = classifier_chain.invoke({"question": question})
    return classification