from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Session
from pydantic import BaseModel
from typing import Optional, List
from databaseHelper.utility import Base


# ORM Model
class Question(Base):
    __tablename__ = "Question"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String, index=True)
    max_words = Column(Integer)
    word_counter = Column(Integer)
    prompt_note = Column(String)
    project_id = Column(Integer, ForeignKey("Project.id"))
    project = relationship("Project", back_populates="questions")


# Pydantic Schemas
class QuestionCreate(BaseModel):
    question: str
    answer: str
    max_words: int
    word_counter: int
    prompt_note: str
    project_id: int


class QuestionResponse(BaseModel):
    id: int
    question: str
    answer: str
    max_words: int
    word_counter: int
    prompt_note: str
    project_id: int

    class Config:
        from_attributes = True


class QuestionUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    max_words: Optional[int] = None
    word_counter: Optional[int] = None
    prompt_note: Optional[str] = None
    project_id: Optional[int] = None


# CRUD Functions
def create_question(db: Session, question: QuestionCreate):
    db_question = Question(
        question=question.question,
        answer=question.answer,
        max_words=question.max_words,
        word_counter=question.word_counter,
        prompt_note=question.prompt_note,
        project_id=question.project_id
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question


def get_questions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Question).offset(skip).limit(limit).all()


def get_question(db: Session, question_id: int):
    return db.query(Question).filter(Question.id == question_id).first()


def delete_question(db: Session, question_id: int):
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if db_question:
        db.delete(db_question)
        db.commit()
    return db_question
