# -*- coding: windows-1251 -*-

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

# Загрузка переменных окружения
load_dotenv()

# API ключ OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Настройка подключения к базе данных

DATABASE_URL = "postgresql://postgres:mamba1705@89.23.112.137/energy_actual"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Настройка логгера
# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Убедитесь, что установлен уровень DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger("app_logger")
