# -*- coding: windows-1251 -*-

import logging
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from customer.config import openai_api_key, logger
from dotenv import load_dotenv

load_dotenv()

# ������ ������������� ��������
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

# ������ ���������� ������ �� ���������
extraction_template = PromptTemplate(
    input_variables=["message"],
    template="""������� ������ � ������� JSON:
    - product_name: �������� ������.
    - quantity: ����������.
    - method: ��������/���������.
    ���������: {message}"""
)

# ������ ��� ����������� ������ �������
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

# ������ ������������� ��������
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
        clean_response = response.strip().lower().replace(".", "").replace(",", "")
        logger.info(f"����� ������: '{clean_response}' ��� ���������: '{message}'")
        return clean_response == "��"
    except Exception as e:
        logger.error(f"������ ��� ���������� �������������: {e}", exc_info=True)
        return False

def determine_stage(state: dict, user_message: str) -> str:
    """
    ���������� ������ ������� �� ������ ��������� ������������.

    Args:
        state (dict): ������� ��������� ������������.
        user_message (str): ��������� �� ������������.

    Returns:
        str: ����������� ������ �������.
    """
    try:
        stage = stage_classifier_chain.invoke({"user_message": user_message}).strip('`')
        state["current_stage"] = stage
        logger.info(f"���������� ������ �������: {stage}")
        return stage
    except Exception as e:
        logger.error(f"������ ��� ����������� ������: {e}", exc_info=True)
        state["current_stage"] = "���������"
        return "���������"

def classify_question(question: str) -> str:
    """
    �������������� ������ ������������.

    Args:
        question (str): ������ ������������.

    Returns:
        str: ��������� �������.
    """
    try:
        logger.info(f"������� ������ � ������ ��� �������������: '{question}'")
        classification = classifier_chain.invoke({"question": question}).strip()
        logger.info(f"������������� �������: {classification}")
        return classification
    except Exception as e:
        logger.error(f"������ ��� ������������� �������: {e}", exc_info=True)
        return "������"
