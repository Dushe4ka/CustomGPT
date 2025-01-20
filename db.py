# -*- coding: windows-1251 -*-

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session
import logging

# Настройки подключения к базе данных
DATABASE_URL = "postgresql://postgres:mamba1705@89.23.112.137/energy_actual"

# Инициализация SQLAlchemy
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db() -> Session:
    """
    Генератор для предоставления сессии базы данных.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_query(query: str, params: dict = None, fetchall: bool = False):
    """
    Выполняет SQL-запрос к базе данных PostgreSQL с использованием SQLAlchemy.

    :param query: SQL-запрос для выполнения.
    :param params: Словарь параметров для безопасного выполнения запроса.
    :param fetchall: Если True, возвращает все строки результата; иначе возвращает одну строку или None.
    :return: Результат запроса (список строк, строка или None).
    """
    result = None
    session = SessionLocal()  # Создаем сессию вручную
    try:
        logging.info(f"Подключение к базе данных...")
        logging.info(f"Выполнение запроса: {query} с параметрами: {params}")
        stmt = text(query)

        # Выполнение SELECT-запросов
        if query.strip().lower().startswith("select"):
            if fetchall:
                result = session.execute(stmt, params).fetchall()
            else:
                result = session.execute(stmt, params).fetchone()
        else:
            # Выполнение запросов на изменение данных
            session.execute(stmt, params)
            session.commit()
            result = True  # Возвращаем True, если изменения выполнены успешно

        logging.info(f"Запрос выполнен успешно, результат: {result}")
        return result
    except Exception as e:
        session.rollback()  # Откатываем изменения в случае ошибки
        logging.error(f"Ошибка при выполнении запроса: {e}", exc_info=True)
        return None
    finally:
        session.close()  # Закрываем сессию
        logging.info("Сессия базы данных закрыта.")
