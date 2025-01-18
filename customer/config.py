# -*- coding: windows-1251 -*-

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

# �������� ���������� ���������
load_dotenv()

# API ���� OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# ��������� ����������� � ���� ������

DATABASE_URL = "postgresql://postgres:mamba1705@89.23.112.137/energy_actual"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ��������� �������
# ��������� �����������
logging.basicConfig(
    level=logging.DEBUG,  # ���������, ��� ���������� ������� DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger("app_logger")
