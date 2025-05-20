from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Session
from pydantic import BaseModel
from typing import Optional, List
from databaseHelper.utility import Base

# ORM Model
class Project(Base):
    __tablename__ = "Project"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    source_text = Column(String, index=True)
    llm_id = Column(Integer, ForeignKey("LLM.id"))

    # New attributes
    api_key = Column(String, index=True)
    prompt_template = Column(String)
    default_max_words = Column(Integer)

    # Relations to other tables
    llm = relationship("LLM", back_populates="projects")
    questions = relationship("Question", back_populates="project", cascade="all, delete-orphan")


# Pydantic Schemas
class ProjectCreate(BaseModel):
    name: str
    source_text: str
    llm_id: int
    api_key: str
    prompt_template: str
    default_max_words: int

class ProjectResponse(BaseModel):
    id: int
    name: str
    source_text: str
    llm_id: int
    api_key: str
    prompt_template: str
    default_max_words: int

    class Config:
        from_attributes = True

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    source_text: Optional[str] = None
    llm_id: Optional[int] = None
    api_key: Optional[str] = None
    prompt_template: Optional[str] = None
    default_max_words: Optional[int] = None


# CRUD Functions
def create_project(db: Session, project: ProjectCreate):
    db_project = Project(
        name=project.name,
        source_text=project.source_text,
        llm_id=project.llm_id,
        api_key=project.api_key,
        prompt_template=project.prompt_template,
        default_max_words=project.default_max_words
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_projects(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Project).offset(skip).limit(limit).all()

def get_project(db: Session, project_id: int):
    return db.query(Project).filter(Project.id == project_id).first()

def delete_project(db: Session, project_id: int):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project:
        db.delete(db_project)
        db.commit()
    return db_project

