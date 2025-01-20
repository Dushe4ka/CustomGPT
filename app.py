# -*- coding: windows-1251 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_sqlalchemy import DBSessionMiddleware
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from starlette.templating import Jinja2Templates

from customer_chat import customer_chat_router


# Создаем приложение FastAPI
app = FastAPI()

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройки базы данных
app.add_middleware(
    DBSessionMiddleware,
    db_url="postgresql://postgres:mamba1705@89.23.112.137/energy_actual",
)

# Добавляем SECRET_KEY (можно использовать в JWT или других механизмах безопасности)
app.state.secret_key = 'mamba676'

# Роутеры
# app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(customer_chat_router, tags=["Customer Chat"])

templates = Jinja2Templates(directory="templates")


@app.get("/kanban", response_class=HTMLResponse)
async def kanban(request: Request):
    return templates.TemplateResponse("kanban.html", {"request": request})

# Запускаем приложение
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
