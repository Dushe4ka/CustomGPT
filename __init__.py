from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db import db  # Импортируем объект базы данных
from fastapi_sqlalchemy import DBSessionMiddleware
from chat import chat_router  # Импортируем роутеры
from customer_chat import customer_chat_router

def create_app():
    # Создаем экземпляр приложения FastAPI
    app = FastAPI()

    # Настройки CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Разрешаем запросы с любых источников
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Настройки приложения
    app.add_middleware(
        DBSessionMiddleware,
        db_url='postgresql://postgres:Art29031705@94.241.173.101/energy_actual',
        secret_key="mamba676",
    )

    # Подключаем маршруты
    app.include_router(chat_router, prefix="/chat", tags=["Chat"])
    app.include_router(customer_chat_router, prefix="/customer_chat", tags=["Customer Chat"])

    return app


# Если файл запускается как основной, запускаем сервер
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:create_app", host="0.0.0.0", port=8000, reload=True)
