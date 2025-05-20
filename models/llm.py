from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship, Session
from pydantic import BaseModel
from typing import List
from databaseHelper.utility import Base

# ORM Model
class LLM(Base):
    __tablename__ = "LLM"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    projects = relationship("Project", back_populates="llm")

# Pydantic Schemas
class LlmCreate(BaseModel):
    name: str

class LlmResponse(BaseModel):
    id: int
    name: str

# CRUD Functions
def create_llm(db: Session, name: str):
    db_llm = LLM(name=name)
    db.add(db_llm)
    db.commit()
    db.refresh(db_llm)
    return db_llm

def get_llms(db: Session, skip: int = 0, limit: int = 100):
    return db.query(LLM).offset(skip).limit(limit).all()
