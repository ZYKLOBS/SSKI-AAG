
from typing import List
from typing import Optional

import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session

from fastapi import FastAPI, Depends, HTTPException

from pydantic import BaseModel

app = FastAPI()

db_url = "sqlite:///./test.db"

engine = create_engine(db_url, connect_args={"check_same_thread": False}) #Allows same connection across different threads

#Autocommit=false so manually commit transactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

llm_names = ["ollama", "claude"]

class LLM(Base):
    __tablename__ = "LLM" #Tablename in db
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    # Add the back-populated relationship to Project
    projects = relationship("Project", back_populates="llm")

class Project(Base):
    __tablename__ = "Project"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    source_text = Column(String, index=True)
    #Add further here, just as proof of concept these 3 shall suffice

    #Foreign key as reference
    llm_id = Column(Integer, ForeignKey("LLM.id"))

    #Make relationships to access LLM
    llm = relationship("LLM", back_populates="projects")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db #yield returns value and then pauses function
    finally: #close database connection after operation
        db.close()


#Project classes

class ProjectCreate(BaseModel):
    name: str
    source_text: str
    llm_id: int

class ProjectResponse(BaseModel):
    id: int
    name: str
    source_text: str
    llm_id: int

    class Config:
        from_attributes = True #originally was orm_mode

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    source_text: Optional[str] = None
    llm_id: Optional[str] = None


# projects Methods


#Create project
@app.post("/projects/", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    if project.llm_id is None:
        print(f"Error: llm_ID may not be None")
        project.llm_id = 1
    if project.llm_id not in range(0+1, len(llm_names)+1): #possibly hardcode values based on llm_names list, increment by 1 since 0 is never primary key for id in LLM Table (sqlite convention)
        print(f"Error: llm_ID outside of range of possible IDs")
        project.llm_id = 1
    db_project = Project(name=project.name, source_text=project.source_text, llm_id=project.llm_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

#Update projects
@app.post("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    db_project.name = project.name if project.name is not None else db_project.name
    db_project.source_text = project.source_text if project.source_text is not None else db_project.source_text
    db_project.llm_id = project.llm_id if project.llm_id is not None else db_project.llm_id

    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/projects/", response_model=List[ProjectResponse])
def read_projects(skip: int = 0, limit: int=100, db: Session = Depends(get_db)):
    """
    :param skip: How many records to skip
    :param limit: How many records to return at most
    :param db:
    :return:
    """
    projects = db.query(Project).offset(skip).limit(limit).all()
    return projects

@app.get("/projects/{project_id}", response_model=ProjectResponse)
def read_project(project_id: int, db: Session = Depends(get_db)):
    """Returns single distinct project based on id
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@app.delete("/projects/{project_id}", response_model=ProjectResponse)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()

    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(db_project)
    db.commit()
    return db_project

#LLM Classes

class llmCreate(BaseModel):
    #Will validate that name is entered and is string
    name: str

class LlmResponse(BaseModel):
    id: int
    name: str

def create_llm(db: Session, name: str):
    db_llm = LLM(name=name)
    db.add(db_llm)
    db.commit()  # Manually commit the transaction to persist the record
    db.refresh(db_llm)  # Refresh the object to get the latest data from the database
    return db_llm



#LLM methods

def insert_llms():
    """
    Manually insert Language Models into DB only if they do not already exist.
    """
    db = SessionLocal()

    try:
        #See above for global parameter llm_names
        for name in llm_names:
            # Check if the LLM already exists
            existing = db.query(LLM).filter_by(name=name).first()
            if not existing:
                new_llm = LLM(name=name)
                db.add(new_llm)

        db.commit()

    except Exception as e:
        print(f"Error inserting LLMs: {e}")
        db.rollback()

    finally:
        db.close()

@app.get("/llms/", response_model=List[LlmResponse])
def read_llms(skip: int = 0, limit: int=100, db: Session = Depends(get_db)):
    """
    :param skip: How many records to skip
    :param limit: How many records to return at most
    :param db:
    :return:
    """
    llms = db.query(LLM).offset(skip).limit(limit).all()
    return llms


if __name__ == "__main__":
    insert_llms()
    uvicorn.run("databaseMain:app", host="127.0.0.1", port=8000, reload=True)
