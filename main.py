import os

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import Form
from fastapi.responses import HTMLResponse, JSONResponse

from sqlalchemy.orm import Session
from typing import List

from databaseHelper.utility import engine, get_db, Base
from models.llm import *
from models.project import *
from models.question import *
from databaseHelper.llm_insert import insert_llms, llm_names

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

Base.metadata.create_all(bind=engine)

class ProjectRequest(BaseModel):
    project_id: int


# Serve homepage index file
@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse(os.path.join(os.getcwd(), "new_index.html"))

@app.post("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db_project.name = project.name if project.name is not None else db_project.name
    db_project.source_text = project.source_text if project.source_text is not None else db_project.source_text
    db_project.llm_id = project.llm_id if project.llm_id is not None else db_project.llm_id
    db_project.api_key = project.api_key if project.api_key is not None else db_project.api_key
    db_project.prompt_template = project.prompt_template if project.prompt_template is not None else db_project.prompt_template
    db_project.default_max_words = project.default_max_words if project.default_max_words is not None else db_project.default_max_words

    db.commit()
    db.refresh(db_project)
    return db_project


#Project methods

@app.post("/projects/", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    if project.llm_id is None:
        print(f"Error: llm_ID may not be None")
        project.llm_id = 1
    if project.llm_id not in range(0+1, len(llm_names)+1):  # Validate LLM ID range
        print(f"Error: llm_ID outside of range of possible IDs")
        project.llm_id = 1

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


@app.post("/load_project/", response_class=HTMLResponse)
async def load_project_by_id(project_id: str = Form(...), db: Session = Depends(get_db)):
    #Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return project.source_text

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

#Keep these requests for now that are not htmx, they are very useful for debugging
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

#For htmx
@app.get("/projects-options/", response_class=HTMLResponse)
def get_project_options(db: Session = Depends(get_db)):
    """Return project options as HTML <option> elements."""
    projects = db.query(Project).all()


    projects_html = "".join(
        f'<option value="{project.id}">{project.name}</option>' for project in projects
    )
    return projects_html



@app.get("/project-source-text/{project_id}", response_class=HTMLResponse)
def project_source_text(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.source_text

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

#LLM Methods
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

#Htmx
@app.get("/models-options/", response_class=HTMLResponse)
def get_llms_options(skip: int = 0, limit: int=100, db: Session = Depends(get_db)):
    """
    :param skip: How many records to skip
    :param limit: How many records to return at most
    :param db:
    :return:
    """
    llms = db.query(LLM).offset(skip).limit(limit).all()

    llm_html = "".join(
        f'<option value="{llm.id}">{llm.name}</option>' for llm in llms
    )
    return llm_html

#Question Methods

@app.post("/questions/", response_model=QuestionResponse)
def create_question(question: QuestionCreate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == question.project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

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


@app.put("/questions/{question_id}", response_model=QuestionResponse)
def update_question(question_id: int, question: QuestionUpdate, db: Session = Depends(get_db)):
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if db_question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    db_question.question = question.question if question.question is not None else db_question.question
    db_question.answer = question.answer if question.answer is not None else db_question.answer
    db_question.max_words = question.max_words if question.max_words is not None else db_question.max_words
    db_question.word_counter = question.word_counter if question.word_counter is not None else db_question.word_counter
    db_question.prompt_note = question.prompt_note if question.prompt_note is not None else db_question.prompt_note
    db_question.project_id = question.project_id if question.project_id is not None else db_question.project_id

    db.commit()
    db.refresh(db_question)
    return db_question

#The first two get methods are more useful for debugging and may be removed later
@app.get("/questions/", response_model=List[QuestionResponse])
def read_questions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    questions = db.query(Question).offset(skip).limit(limit).all()
    return questions

#This is the second debugging get operation
@app.get("/questions/{question_id}", response_model=QuestionResponse)
def read_question(question_id: int, db: Session = Depends(get_db)):
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if db_question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return db_question


# Get all questions for a specific project ID
@app.get("/projects/{project_id}/questions", response_model=List[QuestionResponse])
def read_questions_for_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    questions = db.query(Question).filter(Question.project_id == project_id).all()
    return questions


@app.delete("/questions/{question_id}", response_model=QuestionResponse)
def delete_question(question_id: int, db: Session = Depends(get_db)):
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if db_question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    db.delete(db_question)
    db.commit()
    return db_question

if __name__ == "__main__":
    insert_llms()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
