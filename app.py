# -*- coding: windows-1251 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_sqlalchemy import DBSessionMiddleware
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from starlette.templating import Jinja2Templates

from customer_chat import customer_chat_router


# ������� ���������� FastAPI
app = FastAPI()

# ��������� ����������� �����
app.mount("/static", StaticFiles(directory="static"), name="static")

# ��������� CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ��������� ������� � ����� ����������
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ��������� ���� ������
app.add_middleware(
    DBSessionMiddleware,
    db_url="postgresql://postgres:mamba1705@89.23.112.137/energy_actual",
)

# ��������� SECRET_KEY (����� ������������ � JWT ��� ������ ���������� ������������)
app.state.secret_key = 'mamba676'

# �������
# app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(customer_chat_router, tags=["Customer Chat"])

templates = Jinja2Templates(directory="templates")


@app.get("/kanban", response_class=HTMLResponse)
async def kanban(request: Request):
    return templates.TemplateResponse("kanban.html", {"request": request})

# ��������� ����������
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
