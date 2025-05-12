import os

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.responses import Response

from sqlalchemy.orm import Session
from typing import List

from databaseHelper.utility import engine, get_db, Base
from models.llm import *
from models.project import *
from models.question import *
from databaseHelper.llm_insert import insert_llms, llm_names

from starlette.middleware.sessions import SessionMiddleware

#Change this later
import Ollama
import Claude

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="OBYpXGoIU2be36OktQKxlhauO0aIpr5R")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)
current_id = 1

class ProjectRequest(BaseModel):
    project_id: int

def update_current_id(db: Session):
    # Fetch the first project by ID (smallest ID)
    current_project = db.query(Project).first()

    if current_project:
        current_id = current_project.id
    else:
        current_id = None  # If no project exists


def get_current_project(db: Session):
    global current_id  # If using this, keep it global, but better to pass dynamically
    current_project = db.query(Project).filter(Project.id == current_id).first()
    return current_project

def get_all_projects(db: Session):
    return db.query(Project).all()

def get_all_questions(db: Session):
    return db.query(Question).all()


def get_llm_by_id(db: Session, llm_id: int):
    return db.query(LLM).filter(LLM.id == llm_id).first()


# Serve homepage index file
@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    project_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    projects = db.query(Project).all()
    global current_id
    tmp_cur_id =  update_current_id(db)
    if tmp_cur_id == None:
        tmp_cur_id = 1
    current_id = tmp_cur_id
    project = get_current_project(db)
    if project_id:
        project = get_current_project(db)

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects
        }
    )

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



class ProjectData(BaseModel):
    source_text: str
    prompt_template: str
    api_key: str



#TODO There is a bug, if you add a new question/answer pair and then press save, then the values are not saved

@app.post("/generate-answers/")
async def generate_answers(
        request: Request,
        db: Session = Depends(get_db)
):
    global current_id

    form = await request.form()
    print("Form data:", form)  # Debugging line

    project = get_current_project(db)
    project.source_text = form.get('source_text')
    project.prompt_template = form.get('prompt_template')
    project.api_key = form.get('api_key')
    projects = get_all_projects(db)
    model = form.get("model")
    if int(model) == 1:
        llm = Ollama.Ollama()

    elif int(model) == 2:
        llm = Claude.Claude()

    else:
        print("Something went wrong")

    # Update questions
    index = 0
    questions = db.query(Question).filter(Question.project_id == current_id).all()

    llm.set_source_text(project.source_text)

    for question in questions:
        print(f"Question ID: {question.id},  Question: {question.question}, Answer: {question.answer}, Project_id: {question.project_id}, Model: {model}")

    while True:
        print(f"loop {index}")
        q_question_id = form.get(f'questions[{index}][id]')
        q_question = form.get(f'questions[{index}][question]')
        q_answer = form.get(f'questions[{index}][answer]')
        q_proj_id = form.get(f'questions[{index}][project_id]')
        print(f"\tq_id: {q_question_id}\n\tq_question: {q_question}\n\tq_answer: {q_answer}\n\tq_proj_id: {q_proj_id}")
        if q_question_id is None:
            break  # no more questions
        question = db.query(Question).filter_by(id=q_question_id, project_id=project.id).first()
        if question:
            question.question = q_question
            print(f"Question to generate answer from: {q_question}")
            question.answer = llm.invoke(q_question, project.prompt_template, project.api_key)
            print(f"Generated answer: {question.answer}")

        index += 1

    db.commit()

    project = get_current_project(db)
    # Print the updated questions for debugging
    print("\nUpdated Project Questions:")
    for question in project.questions:
        print(
            f"Question ID: {question.id}, Question: {question.question}, Answer: {question.answer}, Project ID: {question.project_id}")

    print(project.questions)

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects
        }
    )



@app.post("/save-project-button/")
async def save_project_button(request: Request, db: Session = Depends(get_db)):
    try:
        form = await request.form()
        print("Form data:", form)

        project = get_current_project(db)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        project.source_text = form.get('source_text')
        project.prompt_template = form.get('prompt_template')
        project.api_key = form.get('api_key')

        # Update questions
        index = 0
        questions = db.query(Question).filter(Question.project_id == current_id).all()
        for question in questions:
            print(f"Question ID: {question.id},  Question: {question.question}, Answer: {question.answer}, Project_id: {question.project_id}")

        while True:
            q_question_id = form.get(f'questions[{index}][id]')
            q_question = form.get(f'questions[{index}][question]')
            q_answer = form.get(f'questions[{index}][answer]')
            q_proj_id = form.get(f'questions[{index}][project_id]')

            if q_question is None:
                break  # no more questions

            if q_question_id:  # existing question
                question = db.query(Question).filter_by(id=q_question_id, project_id=project.id).first()
                if question:
                    question.question = q_question
                    question.answer = q_answer
            else:  # new question
                new_q = Question(
                    question=q_question,
                    answer=q_answer,
                    project_id=project.id
                )
                db.add(new_q)

            index += 1

        db.commit()
        return JSONResponse(status_code=200, content={"message": "Saved successfully!"})

    except HTTPException as e:
        raise e  # Let FastAPI handle HTTPExceptions as-is

    except Exception as e:
        # Log the error, then return a 500 response
        print(f"Error while saving project: {e}")
        return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})

@app.get("/add_question", response_class=HTMLResponse)
async def add_question(request: Request, db: Session = Depends(get_db)):
    # Generate a new empty question with a unique index
    new_question = Question(question="", answer="", project_id=current_id)
    db.add(new_question)
    db.commit()
    db.refresh(new_question)

    # Ensure the request is included in the template context
    return templates.TemplateResponse("question_block.html", {"request": request, "question": new_question})

#Project methods
@app.get("/Jinja-debug/", response_class=HTMLResponse)
def get_jinja_body_debug(request: Request, db: Session = Depends(get_db)):
    current_project = get_current_project(db)
    return templates.TemplateResponse("main_template.html", {"request": request, "project": current_project})

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

@app.post("/projects-create-button/")
def create_project_button(request: Request, name: str = Form(...), db: Session = Depends(get_db)):
    if not name.strip():
        return {"error": "Project name cannot be empty"}

    # Create new project
    db_project = Project(
        name=name,
        source_text="sauce",
        llm_id=1,
        api_key="sk-3xampl3",
        prompt_template="Default",
        default_max_words=100
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    projects = get_all_projects(db)

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": db_project,
            "projects": projects
        }
    )

@app.post("/projects-rename-button/")
async def rename_project_button(request:Request, rename: str = Form(...), db: Session = Depends(get_db)):
    global current_id
    form = await request.form()
    print("Form data:", form)  # Debugging line

    rename = form.get('rename')

    if not rename.strip():
        return {"error": "Project name cannot be empty"}

    print(rename)
    db_project = get_current_project(db)
    if db_project is None:
        return {"error": f"No project found with id {current_id}"}
    db_project.name = rename if rename is not None else db_project.name
    db.commit()
    db_project = get_current_project(db)
    projects = get_all_projects(db)
    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": db_project,
            "projects": projects
        }
    )

@app.post("/projects-delete-button/")
def delete_project_button(db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == current_id).first()

    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(db_project)
    db.commit()

    # Force a full page reload using HTMX response header
    return Response(status_code=200, headers={"HX-Refresh": "true"})


@app.post("/load-project/", response_class=HTMLResponse)
async def load_project_by_id(
        request: Request,
        project_id: str = Form(...),
        db: Session = Depends(get_db)
):
    global current_id
    current_id = project_id

    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    projects = db.query(Project).all()  # Load all projects for the dropdown

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects
        }
    )


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

#Do at home where you have the current id or a project given, same for safe function
@app.post("/projects-rename/{project_id}", response_model=ProjectResponse)
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
