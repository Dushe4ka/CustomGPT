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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_query(query, params=None, fetchall=False):
    """
    Выполняет SQL-запрос к базе данных PostgreSQL с использованием SQLAlchemy.

    :param query: SQL-запрос для выполнения
    :param params: Параметры для выполнения запроса
    :param fetchall: Если True, возвращает все строки результата; иначе возвращает одну строку или None
    :return: Результат запроса или None в случае ошибок
    """
    result = None
    try:
        logging.info(f"Подключение к базе данных...")
        with SessionLocal() as session:
            logging.info(f"Выполнение запроса: {query} с параметрами: {params}")
            stmt = text(query)

            if query.strip().lower().startswith("select"):
                if fetchall:
                    result = session.execute(stmt, params).fetchall()
                else:
                    result = session.execute(stmt, params).fetchone()
            else:
                session.execute(stmt, params)
                session.commit()
                result = True

            logging.info(f"Запрос выполнен успешно, результат: {result}")
            return result
    except Exception as e:
        logging.error(f"Ошибка при выполнении запроса: {e}")
        return None
