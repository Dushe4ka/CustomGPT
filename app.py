# -*- coding: windows-1251 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_sqlalchemy import DBSessionMiddleware
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from customer_chat import customer_chat_router


# ������� ���������� FastAPI
app = FastAPI()

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
app.include_router(customer_chat_router, prefix="/customer_chat", tags=["Customer Chat"])

# ����������� ����� � �������
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/kanban", response_class=HTMLResponse)
async def kanban():
    # ����� ��������������, ��� ������ kanban.html ��������� � ���������� templates
    with open("/kanban.html", encoding="windows-1251") as file:
        template = Template(file.read())
    return HTMLResponse(template.render())


# ��������� ����������
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
