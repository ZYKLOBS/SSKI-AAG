from fastapi import FastAPI, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from starlette.middleware.sessions import SessionMiddleware

from databaseHelper.utility import engine, get_db
from databaseHelper.llm_insert import insert_llms
from models.llm import *
from models.project import *
from models.question import *

from databaseHelper.llm_insert import llm_names
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
    return current_id

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

    #TODO make this part choosing the model here dynamic
    if int(model) == 1:
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

        project_id = form.get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="Missing project_id")

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        project.source_text = form.get('source_text')
        project.prompt_template = form.get('prompt_template')
        project.api_key = form.get('api_key')

        # Handle questions
        index = 0
        while True:
            q_id = form.get(f'questions[{index}][id]')
            q_question = form.get(f'questions[{index}][question]')
            q_answer = form.get(f'questions[{index}][answer]')

            if q_question is None:
                break

            if q_id:
                question = db.query(Question).filter_by(id=q_id, project_id=project.id).first()
                if question:
                    question.question = q_question
                    question.answer = q_answer
            else:
                new_question = Question(
                    question=q_question,
                    answer=q_answer,
                    project_id=project.id
                )
                db.add(new_question)

            index += 1

        db.commit()
        return JSONResponse(status_code=200, content={"message": "Saved successfully!"})

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error while saving project: {e}")
        return JSONResponse(status_code=500, content={"error": "Unexpected error occurred"})

@app.get("/add_question", response_class=HTMLResponse)
async def add_question(
    request: Request,
    index: int = 0,
    project_id: int = None,
    db: Session = Depends(get_db)
):
    if project_id is None:
        print("ERROR: Project ID is None")
        project = get_current_project(db)
    else:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            # Handle missing project, fallback or error
            raise HTTPException(status_code=404, detail="Project not found")

    new_question = Question(question="", answer="", project_id=project.id)
    db.add(new_question)
    db.commit()
    db.refresh(new_question)

    return templates.TemplateResponse(
        "question_block.html",
        {"request": request, "question": new_question, "index": index}
    )

@app.post("/regenerate_answer/{question_id}", response_class=HTMLResponse)
async def regenerate_answer(request: Request, question_id: int, project_id: int, db: Session = Depends(get_db)):
    form = await request.form()
    print(form)

    project = db.query(Project).filter(Project.id == project_id).first()
    question = db.query(Question).filter_by(id=question_id, project_id=project_id).first()
    projects = get_all_projects(db)


    model = int(form.get("model"))

    if model == 1:
        llm = Claude.Claude()
    else:
        print("Something went wrong")

    llm.set_source_text(project.source_text)
    question.answer = llm.invoke(question.question, project.prompt_template, project.api_key)
    db.commit()

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects
        })

@app.post("/delete_question/{question_id}", response_class=HTMLResponse)
async def delete_question(request: Request, question_id: int, project_id: int, db: Session = Depends(get_db)):
    question = db.query(Question).filter(
        Question.id == question_id,
        Question.project_id == project_id
    ).first()

    if question:
        db.delete(question)
        db.commit()

    project = db.query(Project).filter(Project.id == project_id).first()
    projects = get_all_projects(db)
    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects
        }
    )

#Project methods
@app.get("/Jinja-debug/", response_class=HTMLResponse)
def get_jinja_body_debug(request: Request, db: Session = Depends(get_db)):
    current_project = get_current_project(db)
    return templates.TemplateResponse("main_template.html", {"request": request, "project": current_project})


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



if __name__ == "__main__":
    insert_llms()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
