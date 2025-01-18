# -*- coding: windows-1251 -*-
from models import PredefinedQuestion  # Модель из базы данных
from db import SessionLocal
import os  # Для проверки существования файлов
import faiss  # Для работы с FAISS индексами
import numpy as np  # Для работы с массивами идентификаторов
from langchain_openai import OpenAIEmbeddings  # Для генерации эмбеддингов
from customer.config import SessionLocal, logger, openai_api_key  # Для работы с базой данных, логирование и ключ API
from sqlalchemy.sql import text  # Для выполнения SQL-запросов
from typing import Optional  # Для аннотации возвращаемого типа
from customer.faiss import save_question_index  # Для пересоздания FAISS индекса

def get_response_template(session: SessionLocal, category: str, stage: str) -> Optional[str]:
    """
    Получает шаблон ответа из таблицы predefined_questions на основе категории и стадии диалога.

    :param session: SQLAlchemy сессия для выполнения запросов.
    :param category: Категория вопроса.
    :param stage: Стадия диалога.
    :return: Шаблон ответа или None, если шаблон не найден.
    """
    try:
        logger.info(f"Запрос шаблона для категории '{category}' и стадии '{stage}'.")

        # Выполняем запрос в таблицу predefined_questions
        template = (
            session.query(PredefinedQuestion.answer)
            .filter_by(category=category, stage=stage)
            .scalar()
        )

        if template:
            logger.info(f"Найден шаблон для категории '{category}' и стадии '{stage}'.")
        else:
            logger.warning(f"Шаблон для категории '{category}' и стадии '{stage}' не найден.")

        return template
    except Exception as e:
        logger.error(f"Ошибка при получении шаблона для категории '{category}' и стадии '{stage}': {e}")
        return None

def find_question_template(query: str) -> Optional[str]:
    """
    Находит шаблон ответа на основе вопроса пользователя, используя FAISS.
    """
    logger.info(f"Ищем подходящий шаблон для вопроса: '{query}'")
    try:
        # Проверяем существование файлов
        if not os.path.exists("question_index.faiss") or not os.path.exists("question_ids.npy"):
            logger.info("Индекс вопросов отсутствует. Создаём новый индекс.")
            save_question_index()

        # Чтение индекса и идентификаторов вопросов
        index = faiss.read_index("question_index.faiss")
        question_ids = np.load("question_ids.npy")
        logger.info("Индекс вопросов и идентификаторы успешно загружены.")

        # Генерация эмбеддинга для запроса
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)

        # Поиск ближайших соседей
        D, I = index.search(np.array([query_vector]), k=1)
        logger.info(f"Результаты поиска: D={D}, I={I}")

        # Проверка результата
        if len(I[0]) > 0 and D[0][0] < 0.5:  # Пороговое значение можно настроить
            matched_question_id = question_ids[I[0][0]]
            logger.info(f"Найден подходящий вопрос с ID: {matched_question_id}")

            # Извлекаем шаблон ответа из базы данных
            session = SessionLocal()
            query = text("SELECT answer FROM predefined_questions WHERE id = :id")
            result = session.execute(query, {"id": int(matched_question_id)}).scalar()
            session.close()

            if result:
                logger.info(f"Шаблон ответа: {result}")
                return result
            else:
                logger.warning("Шаблон ответа не найден в базе данных.")
        else:
            logger.warning("Подходящий вопрос не найден в индексе.")
        return None
    except Exception as e:
        logger.error(f"Ошибка при поиске шаблона через FAISS: {e}", exc_info=True)
        return None