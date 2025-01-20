# -*- coding: windows-1251 -*-

import logging
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from customer.config import openai_api_key, logger
from dotenv import load_dotenv

load_dotenv()

# Шаблон классификации вопросов
classification_chain = (
    PromptTemplate.from_template(
        """
        Учитывая следующий вопрос пользователя, определите, относится ли он к товару:
        - Если вопрос связан с товаром (например, вопрос о наличии, цене, характеристиках, доставке товара), ответьте "Да".
        - Если вопрос не связан с товаром (например, общий вопрос о компании, услугах или процессе), ответьте "Нет".
        Вопрос пользователя: {message}
        Ответ:
        """
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)

# Шаблон извлечения данных из сообщения
extraction_template = PromptTemplate(
    input_variables=["message"],
    template="""Извлеки данные в формате JSON:
    - product_name: название товара.
    - quantity: количество.
    - method: доставка/самовывоз.
    Сообщение: {message}"""
)

# Шаблон для определения стадии диалога
stage_classifier_chain = (
    PromptTemplate.from_template(
        """Учитывая сообщение пользователя, определите стадию диалога:
        - `Уточнение остатка`
        - `Уточнение стоимости`
        - `Метод получения`
        - `Непонятно`.

        Сообщение пользователя: {user_message}
        Стадия диалога:"""
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)

# Шаблон классификации вопросов
classifier_chain = (
    PromptTemplate.from_template(
        """Учитывая следующий вопрос пользователя, классифицируйте его как:
        - `наличие товара`,
        - `стоимость товара`,
        - `прочее`.

        Отвечай точно в соответствии с названием категории.

        <вопрос>
        {question}
        </вопрос>

        Классификация:"""
    )
    | ChatOpenAI(api_key=openai_api_key)
    | StrOutputParser()
)

def is_product_related(message: str) -> bool:
    """
    Определяет, связан ли вопрос пользователя с товаром.

    Args:
        message (str): Сообщение пользователя.

    Returns:
        bool: True, если вопрос связан с товаром, иначе False.
    """
    try:
        logger.info(f"Передаём сообщение в модель для классификации: '{message}'")
        response = classification_chain.invoke({"message": message})
        clean_response = response.strip().lower().replace(".", "").replace(",", "")
        logger.info(f"Ответ модели: '{clean_response}' для сообщения: '{message}'")
        return clean_response == "да"
    except Exception as e:
        logger.error(f"Ошибка при выполнении классификации: {e}", exc_info=True)
        return False

def determine_stage(state: dict, user_message: str) -> str:
    """
    Определяет стадию диалога на основе сообщения пользователя.

    Args:
        state (dict): Текущее состояние пользователя.
        user_message (str): Сообщение от пользователя.

    Returns:
        str: Определённая стадия диалога.
    """
    try:
        stage = stage_classifier_chain.invoke({"user_message": user_message}).strip('`')
        state["current_stage"] = stage
        logger.info(f"Определена стадия диалога: {stage}")
        return stage
    except Exception as e:
        logger.error(f"Ошибка при определении стадии: {e}", exc_info=True)
        state["current_stage"] = "Непонятно"
        return "Непонятно"

def classify_question(question: str) -> str:
    """
    Классифицирует вопрос пользователя.

    Args:
        question (str): Вопрос пользователя.

    Returns:
        str: Категория вопроса.
    """
    try:
        logger.info(f"Передаём вопрос в модель для классификации: '{question}'")
        classification = classifier_chain.invoke({"question": question}).strip()
        logger.info(f"Классификация вопроса: {classification}")
        return classification
    except Exception as e:
        logger.error(f"Ошибка при классификации вопроса: {e}", exc_info=True)
        return "прочее"
