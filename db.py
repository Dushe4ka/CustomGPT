# -*- coding: windows-1251 -*-

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session
import logging

# ��������� ����������� � ���� ������
DATABASE_URL = "postgresql://postgres:mamba1705@89.23.112.137/energy_actual"

# ������������� SQLAlchemy
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
    ��������� SQL-������ � ���� ������ PostgreSQL � �������������� SQLAlchemy.

    :param query: SQL-������ ��� ����������
    :param params: ��������� ��� ���������� �������
    :param fetchall: ���� True, ���������� ��� ������ ����������; ����� ���������� ���� ������ ��� None
    :return: ��������� ������� ��� None � ������ ������
    """
    result = None
    try:
        logging.info(f"����������� � ���� ������...")
        with SessionLocal() as session:
            logging.info(f"���������� �������: {query} � �����������: {params}")
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

            logging.info(f"������ �������� �������, ���������: {result}")
            return result
    except Exception as e:
        logging.error(f"������ ��� ���������� �������: {e}")
        return None
