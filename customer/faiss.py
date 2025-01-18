# -*- coding: windows-1251 -*-
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from sqlalchemy.sql import text
import os  # Добавьте этот импорт


import faiss
from customer.config import SessionLocal, openai_api_key, logger
load_dotenv()

# Создание эмбеддингов из базы данных
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

# Создание FAISS индекса
def save_faiss_index():
    product_ids, product_vectors = create_embeddings_from_db()
    dimension = len(product_vectors[0])  # Размерность эмбеддинга
    index = faiss.IndexFlatL2(dimension)
    index.add(product_vectors)

    # Сохраняем индекс и соответствие product_ids
    faiss.write_index(index, "product_index.faiss")
    np.save("product_ids.npy", np.array(product_ids))

# Поиск товара в FAISS
def find_product_with_faiss(query: str):
    logger.info(f"Ищем товар по запросу: '{query}'")
    try:
        # Проверяем существование файлов
        if not os.path.exists("product_index.faiss") or not os.path.exists("product_ids.npy"):
            logger.info("FAISS индекс отсутствует. Создаём новый индекс.")
            save_faiss_index()

        # Чтение индекса и идентификаторов товаров
        index = faiss.read_index("product_index.faiss")
        product_ids = np.load("product_ids.npy")
        logger.info("FAISS индекс и идентификаторы успешно загружены.")

        # Генерация эмбеддинга для запроса
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logger.info("Эмбеддинг для запроса успешно создан.")

        # Поиск ближайших соседей
        D, I = index.search(np.array([query_vector]), k=1)
        logger.info(f"Результаты поиска: D={D}, I={I}")

        # Проверка результата
        if len(I[0]) > 0 and D[0][0] < 0.5:
            matched_product_id = product_ids[I[0][0]]
            logger.info(f"Найден товар с ID: {matched_product_id}, расстояние: {D[0][0]}")
            return matched_product_id
        else:
            logger.warning("Товар не найден в FAISS.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при поиске товара в FAISS: {e}", exc_info=True)
        return None

def get_product_by_id(product_id: int):
    """
    Извлекает информацию о товаре из базы данных по его ID.

    Args:
        product_id (int): Уникальный идентификатор товара.

    Returns:
        dict: Информация о товаре с ключами 'product_name', 'availability' и 'price'.
        None: Если товар не найден или произошла ошибка.
    """
    logger.info(f"Извлекаем информацию о товаре с ID: {product_id}")
    session = SessionLocal()
    try:
        # Убедимся, что product_id имеет правильный тип
        product_id = int(product_id)

        # Выполняем запрос к базе данных
        query = text("SELECT name, quantity, price FROM base_product WHERE id = :id")
        product = session.execute(query, {"id": product_id}).fetchone()

        # Проверяем результат
        if product:
            product_data = {
                "product_name": product[0],
                "availability": product[1],
                "price": product[2]
            }
            logger.info(f"Информация о товаре: {product_data}")
            return product_data
        else:
            logger.warning(f"Товар с ID {product_id} не найден в базе данных.")
            return None
    except ValueError as ve:
        logger.error(f"Неверный тип данных для product_id: {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении информации о товаре: {e}", exc_info=True)
        return None
    finally:
        # Закрываем сессию в любом случае
        session.close()

def find_product_in_faiss(query: str) -> dict:
    logger.info(f"Ищем товар в FAISS по запросу: '{query}'")
    try:
        # Поиск товара в FAISS
        product_id = find_product_with_faiss(query)
        if product_id:
            # Получение информации о товаре из базы данных
            product = get_product_by_id(product_id)
            if product:
                logger.info(f"Товар успешно найден: {product}")
                return product
        logger.warning("Товар не найден через FAISS.")
        return None
    except Exception as e:
        logger.error(f"Ошибка при поиске товара через FAISS: {e}", exc_info=True)
        return None

def create_question_embeddings():
    """
    Создаёт эмбеддинги для вопросов из таблицы predefined_questions.
    """
    session = SessionLocal()
    try:
        # Извлекаем вопросы и их ID из таблицы predefined_questions
        questions = session.execute(text("SELECT id, question FROM predefined_questions")).fetchall()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        question_ids = []
        question_vectors = []

        for question_id, question_text in questions:
            if question_text:  # Проверяем, что текст вопроса не пустой
                vector = embeddings.embed_query(question_text)
                question_ids.append(question_id)
                question_vectors.append(vector)

        return question_ids, np.array(question_vectors)
    finally:
        session.close()


def save_question_index():
    """
    Создаёт и сохраняет FAISS индекс для вопросов.
    """
    question_ids, question_vectors = create_question_embeddings()
    dimension = len(question_vectors[0])  # Размерность эмбеддингов
    index = faiss.IndexFlatL2(dimension)
    index.add(question_vectors)

    # Сохраняем индекс и соответствие question_ids
    faiss.write_index(index, "question_index.faiss")
    np.save("question_ids.npy", np.array(question_ids))