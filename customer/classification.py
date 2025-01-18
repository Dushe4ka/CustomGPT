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



# Шаблон классификации
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

extraction_template = PromptTemplate(
    input_variables=["message"],
    template="""Извлеки данные в формате JSON:
    - product_name: название товара.
    - quantity: количество.
    - method: доставка/самовывоз.
    Сообщение: {message}"""
)

#классификация стадии диалога
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

# Создаём цепочку классификации
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
        logger.info(f"Ответ модели: '{response.strip()}' для сообщения: '{message}'")

        # Очистка ответа модели
        clean_response = response.strip().lower().replace(".", "").replace(",", "")
        logger.info(f"Очищенный ответ модели: '{clean_response}'")

        if clean_response not in ["да", "нет"]:
            logger.warning(f"Неопределённый ответ модели: '{clean_response}'. Считаем, что вопрос не связан с товаром.")
            return False
        return clean_response == "да"
    except Exception as e:
        logger.error(f"Ошибка при выполнении классификации: {e}", exc_info=True)
        return False


#Обновление состояния пользователя
def determine_stage(state: dict, user_message: str) -> str:
    """
    Определяет стадию диалога на основе сообщения пользователя.

    :param state: Текущее состояние пользователя.
    :param user_message: Сообщение от пользователя.
    :return: Определённая стадия диалога.
    """
    try:
        stage = stage_classifier_chain.invoke({"user_message": user_message}).strip()
        stage = stage.strip('`')  # Удаляем лишние обратные апострофы
        state["current_stage"] = stage
        logger.info(f"Очищенная стадия диалога: {stage}")
        return stage
    except Exception as e:
        logger.error(f"Ошибка при определении стадии: {e}")
        state["current_stage"] = "Непонятно"
        return "Непонятно"

# Классификация вопроса
def classify_question(question):
    classification = classifier_chain.invoke({"question": question})
    return classification