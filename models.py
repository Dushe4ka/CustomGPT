# -*- coding: windows-1251 -*-

from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from db import SessionLocal

# Модель для таблицы категорий
class Category:
    __tablename__ = 'categories'

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)

    # Связь с таблицей вопросов и ответов
    questions_answers = relationship('QuestionAnswer', back_populates='category', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Category(id={self.category_id}, name={self.name})>"

# Модель для таблицы вопросов и ответов
class QuestionAnswer:
    __tablename__ = 'questions_answers'

    question_answer_id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey('categories.category_id', ondelete='CASCADE'), nullable=False)
    key_info = Column(String(255), nullable=True)  # Добавлено поле key_info

    # Связь с категорией
    category = relationship('Category', back_populates='questions_answers')

    def __repr__(self):
        return f"<QuestionAnswer(id={self.question_answer_id}, question={self.question}, answer={self.answer})>"

# Модель для таблицы компаний
class Company:
    __tablename__ = 'companies'

    company_id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(255), nullable=False)

    def __repr__(self):
        return f"<Company(id={self.company_id}, name={self.company_name})>"

# Модель для отслеживания состояния диалога пользователя
class UserDialogState:
    __tablename__ = 'user_dialog_state'

    state_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=True)
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=True)

    mentioned_availability = Column(Boolean, default=False)
    mentioned_price = Column(Boolean, default=False)
    determined_delivery_method = Column(Boolean, default=False)

    user = relationship('User', back_populates='dialog_state')

    def __repr__(self):
        return (f"<UserDialogState(state_id={self.state_id}, session_id={self.session_id}, "
                f"mentioned_availability={self.mentioned_availability}, mentioned_price={self.mentioned_price}, "
                f"determined_delivery_method={self.determined_delivery_method})>")

# Модель для пользователей
class User:
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False, unique=True)

    dialog_state = relationship('UserDialogState', back_populates='user', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username})>"

# Модель для клиентов
class Customer:
    __tablename__ = 'customers'

    customer_id = Column(Integer, primary_key=True, autoincrement=True)
    last_name = Column(String(100), nullable=False)
    first_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20), nullable=False)

    change_logs = relationship('CustomerChangeLog', back_populates='customer', cascade="all, delete-orphan")
    deals = relationship('Deal', back_populates='customer', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Customer(customer_id={self.customer_id}, last_name={self.last_name}, first_name={self.first_name}, email={self.email})>"

# Модель истории изменений клиентов
class CustomerChangeLog:
    __tablename__ = 'customer_change_log'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False)
    field_name = Column(String(100), nullable=False)
    old_value = Column(String(255), nullable=True)
    new_value = Column(String(255), nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship('Customer', back_populates='change_logs')

# Модель для сделок
class Deal:
    __tablename__ = 'deals'

    deal_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False)
    stage_id = Column(Integer, ForeignKey('stages_deal.stage_id', ondelete='SET NULL'), nullable=True)
    name = Column(String(255), nullable=False)
    price = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship('Customer', back_populates='deals')
    stage = relationship('StageDeal', back_populates='deals')

# Модель для стадий сделок
class StageDeal:
    __tablename__ = 'stages_deal'

    stage_id = Column(Integer, primary_key=True, autoincrement=True)
    stage_name = Column(String(255), nullable=False)

    deals = relationship('Deal', back_populates='stage', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<StageDeal(stage_id={self.stage_id}, stage_name={self.stage_name})>"

class Interaction:
    __tablename__ = 'interactions'

    # Определение колонок таблицы
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

    def __repr__(self):
        return f"<Interaction {self.id} - User: {self.user_id}>"

# Модель для предопределённых вопросов
class PredefinedQuestion:
    __tablename__ = 'predefined_questions'

    # Определение колонок таблицы
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(255), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    stage = Column(String(50))  # Добавляем новое поле

    def __repr__(self):
        return (
            f"<PredefinedQuestion(id={self.id}, category={self.category}, "
            f"question={self.question}, answer={self.answer}, stage={self.stage})>"
        )
