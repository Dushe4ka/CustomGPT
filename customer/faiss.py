# -*- coding: windows-1251 -*-
import os
import numpy as np
import faiss
from sqlalchemy.sql import text
from customer.config import SessionLocal, openai_api_key, logger
from langchain_openai import OpenAIEmbeddings

# Создание эмбеддингов из базы данных
def create_embeddings_from_db():
    session = SessionLocal()
    try:
        products = session.execute(text("SELECT id, name FROM base_product")).fetchall()
        if not products:
            logger.warning("В базе данных нет товаров для создания эмбеддингов.")
            return [], np.array([])

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        product_ids = []
        product_vectors = []

        for product_id, product_name in products:
            if product_name:  # Проверяем, что имя товара не пустое
                vector = embeddings.embed_query(product_name)
                product_ids.append(product_id)
                product_vectors.append(vector)

        return product_ids, np.array(product_vectors)
    finally:
        session.close()

# Создание FAISS индекса
def save_faiss_index():
    product_ids, product_vectors = create_embeddings_from_db()
    if len(product_vectors) == 0:
        logger.warning("Индекс FAISS не был создан: векторные данные отсутствуют.")
        return

    dimension = len(product_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(product_vectors)

    # Сохраняем индекс и идентификаторы
    faiss.write_index(index, "product_index.faiss")
    np.save("product_ids.npy", np.array(product_ids))
    logger.info("FAISS индекс и идентификаторы успешно сохранены.")

# Поиск товара в FAISS
def find_product_with_faiss(query: str):
    logger.info(f"Ищем товар по запросу: '{query}'")
    try:
        # Проверяем существование файлов
        if not os.path.exists("product_index.faiss") or not os.path.exists("product_ids.npy"):
            logger.info("FAISS индекс отсутствует. Создаём новый индекс.")
            save_faiss_index()

        # Чтение индекса и идентификаторов
        index = faiss.read_index("product_index.faiss")
        product_ids = np.load("product_ids.npy", allow_pickle=True)
        logger.info("FAISS индекс и идентификаторы успешно загружены.")

        # Генерация эмбеддинга для запроса
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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

# Получение информации о товаре
def get_product_by_id(product_id: int):
    logger.info(f"Извлекаем информацию о товаре с ID: {product_id}")
    session = SessionLocal()
    try:
        product_id = int(product_id)  # Проверка типа
        query = text("SELECT name, quantity, price FROM base_product WHERE id = :id")
        product = session.execute(query, {"id": product_id}).fetchone()

        if product:
            product_data = {
                "product_name": product[0],
                "availability": product[1],
                "price": product[2]
            }
            logger.info(f"Информация о товаре: {product_data}")
            return product_data
        else:
            logger.warning(f"Товар с ID {product_id} не найден.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при получении информации о товаре: {e}", exc_info=True)
        return None
    finally:
        session.close()

# Поиск товара в FAISS
def find_product_in_faiss(query: str) -> dict:
    logger.info(f"Ищем товар в FAISS по запросу: '{query}'")
    try:
        product_id = find_product_with_faiss(query)
        if product_id:
            product = get_product_by_id(product_id)
            if product:
                logger.info(f"Товар успешно найден: {product}")
                return product
        logger.warning("Товар не найден через FAISS.")
        return None
    except Exception as e:
        logger.error(f"Ошибка при поиске товара через FAISS: {e}", exc_info=True)
        return None

# Создание и сохранение эмбеддингов для вопросов
def create_question_embeddings():
    session = SessionLocal()
    try:
        questions = session.execute(text("SELECT id, question FROM predefined_questions")).fetchall()
        if not questions:
            logger.warning("В базе данных нет вопросов для создания эмбеддингов.")
            return [], np.array([])

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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
    question_ids, question_vectors = create_question_embeddings()
    if len(question_vectors) == 0:
        logger.warning("Индекс вопросов не был создан: векторные данные отсутствуют.")
        return

    dimension = len(question_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(question_vectors)

    # Сохраняем индекс и идентификаторы
    faiss.write_index(index, "question_index.faiss")
    np.save("question_ids.npy", np.array(question_ids))
    logger.info("FAISS индекс для вопросов успешно сохранён.")
