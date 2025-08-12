from io import BytesIO
import io

import time

from fastapi import FastAPI, Depends, Request, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from numpy.f2py.crackfortran import sourcecodeform

from sqlalchemy.exc import IntegrityError


from starlette.middleware.sessions import SessionMiddleware

import OpenAI
from databaseHelper.utility import engine, get_db
from databaseHelper.llm_insert import insert_llms
from models.llm import *
from models.project import *
from models.question import *

import pandas as pd


from databaseHelper.llm_insert import llm_names
import Ollama
import Claude
from anthropic import AuthenticationError

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

@app.post("/set-model/")
def set_model(
    project_id: int = Form(...),
    model: int = Form(...),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return {"error": "Project not found"}

    project.model = model  # Update the model column
    db.commit()           # Commit to save changes

    return {"message": "Model updated successfully"}

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

@app.post("/upload_excel")
async def upload_excel(
    request: Request,
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    error_message = None
    db_project = None

    if file is None or file.filename == "":
        error_message = "No file was uploaded."
    else:
        try:
            contents = await file.read()
            df = pd.read_excel(BytesIO(contents), header=None)

            # Expecting: meta, question, answer
            df.columns = ['Meta', 'question', 'answer']

            df_A = df['Meta']
            df_B = df['question']
            df_C = df['answer']

            # Check project name
            if pd.isna(df_A.iloc[0]) or str(df_A.iloc[0]).strip() == "":
                error_message = "Project name cannot be empty."
            else:
                # Create project
                db_project = Project(
                    name=str(df_A[0]).strip(),
                    source_text=str(df_A[1]) if len(df_A) > 1 else "",
                    llm_id=1,
                    api_key="",
                    prompt_template=str(df_A[2]) if len(df_A) > 2 else "Default",
                    default_max_words=100
                )
                try:
                    db.add(db_project)
                    db.commit()
                    db.refresh(db_project)
                except IntegrityError:
                    db.rollback()
                    error_message = "A project with this name already exists."
                    db_project = get_current_project(db)

                # Insert questions if no error so far
                if error_message is None:
                    for question, answer in zip(df_B, df_C):
                        if pd.isna(question):
                            continue
                        db_question = Question(
                            question=str(question).strip(),
                            answer=str(answer).strip() if not pd.isna(answer) else "",
                            project_id=db_project.id
                        )
                        db.add(db_question)
                    db.commit()

        except Exception as e:
            # Only catch unexpected errors now
            error_message = f"Failed to read Excel file: {str(e)}"

    db_project = db_project or get_current_project(db)
    projects = get_all_projects(db)

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": db_project,
            "projects": projects,
            "error_message": error_message
        }
    )

@app.get("/export_excel")
def export_excel(request: Request, db: Session = Depends(get_db)):
    current_project = get_current_project(db)
    if not current_project:
        return JSONResponse(status_code=404, content={"error": "No current project selected"})

    questions = db.query(Question).filter(Question.project_id == current_project.id).all()

    # Prepare column A with project meta info
    col_a = [
        current_project.name,
        current_project.source_text,
        current_project.prompt_template,
    ]

    num_questions = len(questions)
    # Pad col_a to have length at least equal to number of questions
    if num_questions > 3:
        col_a.extend([""] * (num_questions - 3))

    # Prepare columns B and C from questions
    col_b = [q.question for q in questions]
    col_c = [q.answer for q in questions]

    # Pad col_b and col_c if needed (e.g. if fewer than 3 questions)
    min_len = max(len(col_a), len(col_b), len(col_c))
    if len(col_b) < min_len:
        col_b.extend([""] * (min_len - len(col_b)))
    if len(col_c) < min_len:
        col_c.extend([""] * (min_len - len(col_c)))

    # Now all columns are same length, create dataframe
    df = pd.DataFrame({
        'A': col_a,
        'B': col_b,
        'C': col_c,
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, header=False)
    output.seek(0)

    filename = f"{current_project.name}.xlsx"

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
@app.post("/generate-answers/")
async def generate_answers(request: Request, db: Session = Depends(get_db)):
    start_time = time.perf_counter()
    print(f"[DEBUG 0] Start /generate-answers at {start_time:.4f}")

    error_message = None
    form = await request.form()
    print(f"[DEBUG 1] Received form in {time.perf_counter() - start_time:.4f}s")

    # Get project_id
    project_id = form.get("project_id")
    if not project_id:
        print(f"[DEBUG] Missing project_id after {time.perf_counter() - start_time:.4f}s")
        return JSONResponse(status_code=400, content={"error": "Missing project_id"})

    project = db.query(Project).filter(Project.id == project_id).first()
    print(f"[DEBUG 2] Queried project in {time.perf_counter() - start_time:.4f}s")

    if not project:
        print(f"[DEBUG] Project not found in {time.perf_counter() - start_time:.4f}s")
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    # Update project fields
    project.source_text = form.get('source_text')
    project.prompt_template = form.get('prompt_template')
    api_key = form.get('api_key')  # transient use only
    print(f"[DEBUG 3] Updated project fields in {time.perf_counter() - start_time:.4f}s")

    # Update or add questions
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
            db.add(Question(question=q_question, answer=q_answer, project_id=project.id))

        index += 1

    print(f"[DEBUG 4] Processed {index} questions in {time.perf_counter() - start_time:.4f}s")

    model = form.get("model")
    if not model or not model.isdigit():
        print(f"[DEBUG] Invalid model after {time.perf_counter() - start_time:.4f}s")
        return JSONResponse(status_code=400, content={"error": "Invalid or missing model selected"})
    project.llm_id = int(model)

    db.commit()
    print(f"[DEBUG 5] Saved basic project changes in {time.perf_counter() - start_time:.4f}s")

    # Model selection
    model_int = int(model)
    llm = None

    if model_int == 1:
        if not api_key or not api_key.strip():
            error_message = "API key cannot be empty. Please enter a valid API key."
        else:
            llm = Claude.Claude()
    elif model_int == 2:
        if not api_key or not api_key.strip():
            error_message = "API key cannot be empty. Please enter a valid API key."
        else:
            llm = OpenAI.OpenAIWrapper()
    elif model_int == 3:
        llm = Ollama.OllamaWrapper()
    else:
        error_message = "Invalid model selected."
        projects = get_all_projects(db)
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message
            }
        )

    print(f"[DEBUG 6] Model selection done in {time.perf_counter() - start_time:.4f}s")

    if llm:
        llm.set_source_text(project.source_text)
        questions = db.query(Question).filter(Question.project_id == project.id).all()
        print(f"[DEBUG 7] Loaded {len(questions)} questions in {time.perf_counter() - start_time:.4f}s")

        try:
            for idx, question in enumerate(questions, start=1):
                try:
                    t0 = time.perf_counter()
                    question.answer = llm.invoke(question.question, project.prompt_template, api_key)
                    print(f"[DEBUG 8.{idx}] Answer generated in {time.perf_counter() - t0:.4f}s")
                except AuthenticationError:
                    question.answer = "[AUTHENTICATION ERROR: Invalid API key]"
                    error_message = "Invalid API key. Please check it and try again."
                except Exception as e:
                    question.answer = "[ERROR generating answer]"
                    error_message = f"Error generating answer for Q{idx}: {e}"
            db.commit()
            print(f"[DEBUG 9] All answers generated in {time.perf_counter() - start_time:.4f}s")
        except Exception as e:
            error_message = f"Unexpected error during answer generation: {e}"

    projects = get_all_projects(db)
    total_time = time.perf_counter() - start_time
    print(f"[DEBUG END] Request completed in {total_time:.4f}s")

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "error_message": error_message
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
        #Dont save since we want each user to only have access to a key themselves
        #project.api_key = form.get('api_key')
        project.llm_id = int(form.get('model'))


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

@app.post("/delete_question/{question_id}", response_class=HTMLResponse)
async def delete_question(
    request: Request,
    question_id: int,
    project_id: int,
    db: Session = Depends(get_db)
):
    try:
        form = await request.form()
        print("Form data:", form)

        # Load project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Save project fields
        project.source_text = form.get('source_text')
        project.prompt_template = form.get('prompt_template')
        project.api_key = form.get('api_key')

        # Save question updates (excluding the one to be deleted)
        index = 0
        while True:
            q_id = form.get(f'questions[{index}][id]')
            q_question = form.get(f'questions[{index}][question]')
            q_answer = form.get(f'questions[{index}][answer]')

            if q_question is None:
                break

            if str(q_id) != str(question_id):  # skip the one being deleted
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

        # Now delete the question
        to_delete = db.query(Question).filter(
            Question.id == question_id,
            Question.project_id == project_id
        ).first()
        if to_delete:
            db.delete(to_delete)

        db.commit()

        # Reload data for rendering
        projects = get_all_projects(db)
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "questions": project.questions
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during delete/save: {e}")
        return JSONResponse(status_code=500, content={"error": "Unexpected error occurred"})
@app.post("/regenerate_answer/{question_id}", response_class=HTMLResponse)
async def regenerate_answer(request: Request, question_id: int, project_id: int, db: Session = Depends(get_db)):
    form = await request.form()
    print("Regenerate Form:", form)

    # Load project from DB
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update project fields EXCEPT api_key (do not save api_key to DB)
    project.source_text = form.get('source_text')
    project.prompt_template = form.get('prompt_template')
    # project.api_key = form.get('api_key')  # <-- REMOVE this line to prevent saving API key

    # Update questions from form
    index = 0
    while True:
        q_id = form.get(f'questions[{index}][id]')
        q_question = form.get(f'questions[{index}][question]')
        q_answer = form.get(f'questions[{index}][answer]')

        if q_question is None:
            break

        if q_id:
            question_db = db.query(Question).filter_by(id=q_id, project_id=project.id).first()
            if question_db:
                question_db.question = q_question
                question_db.answer = q_answer
        else:
            new_question = Question(
                question=q_question,
                answer=q_answer,
                project_id=project.id
            )
            db.add(new_question)
        index += 1

    db.commit()

    # Reload question from DB to get latest
    question = db.query(Question).filter_by(id=question_id, project_id=project_id).first()
    projects = get_all_projects(db)
    error_message = None

    if not question or not question.question.strip():
        error_message = "Can't regenerate answer: question is empty."
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message,
            }
        )

    # Select model
    model_str = form.get("model", "0")
    try:
        model = int(model_str)
    except ValueError:
        model = 0

    project.llm_id = model

    if model == 1:
        llm = Claude.Claude()
    elif model == 2:
        llm = OpenAI.OpenAIWrapper()
    elif model == 3:
        llm = Ollama.OllamaWrapper()
    else:
        error_message = "Invalid model selected."
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message,
            }
        )

    # Use API key dynamically from form, do NOT save it to DB or send it back
    api_key = form.get("api_key", "").strip()
    if not api_key:
        error_message = "API key cannot be empty. Please enter a valid API key."
        question.answer = "[AUTHENTICATION ERROR: API key is empty]"
    else:
        llm.set_source_text(project.source_text)
        try:
            question.answer = llm.invoke(question.question, project.prompt_template, api_key)
        except AuthenticationError as e:
            print(f"API key authentication failed: {e}")
            question.answer = "[AUTHENTICATION ERROR: Invalid API key]"
            error_message = "Invalid API key. Please check it and try again."
        except Exception:
            question.answer = "[AUTHENTICATION ERROR: Invalid API key or Ollama not running]"
            error_message = "Invalid API key or Ollama server not running."

    db.commit()

    # Render template WITHOUT passing the API key back
    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "error_message": error_message,
        }
    )


@app.post("/refine_answer/{question_id}", response_class=HTMLResponse)
async def refine_answer(
        request: Request,
        question_id: int,
        project_id: int = Query(...),
        db: Session = Depends(get_db)
):
    form = await request.form()
    print("Refine Form:", form)

    # Load project from DB
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update project fields EXCEPT api_key (do NOT save api_key)
    project.source_text = form.get('source_text')
    project.prompt_template = form.get('prompt_template')
    # Do NOT save api_key to DB
    # project.api_key = form.get('api_key')

    # Get API key from form only for LLM call
    api_key = form.get('api_key', '').strip()

    # Collect refine prompts (list)
    refine_prompts = form.getlist("refine_prompt")

    index = 0
    refine_prompt_used = ""

    # Update questions from form
    while True:
        q_id = form.get(f'questions[{index}][id]')
        q_question = form.get(f'questions[{index}][question]')
        q_answer = form.get(f'questions[{index}][answer]')

        if q_question is None:
            break

        if q_id:
            question_obj = db.query(Question).filter_by(id=q_id, project_id=project.id).first()
            if question_obj:
                question_obj.question = q_question
                question_obj.answer = q_answer
            # Match question to refine and get corresponding refine prompt by index
            if int(q_id) == question_id and index < len(refine_prompts):
                refine_prompt_used = refine_prompts[index].strip()
        else:
            new_question = Question(
                question=q_question,
                answer=q_answer,
                project_id=project.id
            )
            db.add(new_question)

        index += 1

    db.commit()

    # Reload question from DB to get latest
    question = db.query(Question).filter_by(id=question_id, project_id=project_id).first()
    projects = get_all_projects(db)
    error_message = None

    # Validate question and previous answer
    if not question or not question.question.strip() or not question.answer.strip():
        error_message = "Can't refine: Question or previous answer is empty."
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message,
            }
        )

    # Check API key presence (do NOT save it)
    if not api_key:
        error_message = "API key cannot be empty. Please enter a valid API key."
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message,
            }
        )

    # Model selection, fallback to 1 if invalid
    try:
        model = int(form.get("model", 1))
        project.llm_id = model
    except (ValueError, TypeError):
        model = 1

    if model == 1:
        llm = Claude.Claude()
    elif model == 2:
        llm = OpenAI.OpenAIWrapper()
    elif model == 3:
        llm = Ollama.OllamaWrapper()
    else:
        error_message = "Invalid model selected."
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "error_message": error_message,
            }
        )

    llm.set_source_text(project.source_text)
    print(f"Refine prompt used: {refine_prompt_used}")

    # Call LLM refine with API key from form only
    try:
        question.answer = llm.refine(
            user_prompt=question.question,
            refine_prompt=refine_prompt_used,
            previous_answer=question.answer,
            prompt_template=project.prompt_template,
            api_key=api_key
        )
    except AuthenticationError as e:
        print(f"API key authentication failed: {e}")
        question.answer = "[AUTHENTICATION ERROR: Invalid API key]"
        error_message = "Invalid API key. Please check it and try again."
    except Exception:
        question.answer = "[AUTHENTICATION ERROR: Invalid API key or Ollama not running]"
        error_message = "Invalid API key or Ollama Server not running."

    db.commit()

    # Render template WITHOUT api_key in context to keep frontend input persistent
    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "error_message": error_message,
        }
    )


#Project methods
@app.get("/Jinja-debug/", response_class=HTMLResponse)
def get_jinja_body_debug(request: Request, db: Session = Depends(get_db)):
    current_project = get_current_project(db)
    return templates.TemplateResponse("main_template.html", {"request": request, "project": current_project})

@app.post("/projects-create-button/")
def create_project_button(request: Request, name: Optional[str] = Form(""), db: Session = Depends(get_db)):
    global current_id

    error_message = None
    db_project = None  #

    if not name.strip():
        projects = get_all_projects(db)
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "error_message": "Project name cannot be empty.",
                "projects": projects,
                "project": get_current_project(db)
            }
        )

    db_project = Project(
        name=name,
        source_text="sauce",
        llm_id=1,
        api_key="sk-3xampl3",
        prompt_template="Default",
        default_max_words=100
    )

    try:
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        current_id = db_project.id
    except IntegrityError:
        db.rollback()
        error_message = "duplicate project name."
        db_project = get_current_project(db)

    projects = get_all_projects(db)
    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": db_project,
            "projects": projects,
            "error_message": error_message
        }
    )
@app.post("/projects-rename-button/")
async def rename_project_button(
    request: Request,
    rename: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    global current_id
    error_message = None

    if not rename.strip():
        projects = get_all_projects(db)
        return templates.TemplateResponse(
            "main_template.html",
            {
                "request": request,
                "error_message": "Project name cannot be empty.",
                "project": get_current_project(db),
                "projects": projects
            }
        )

    db_project = get_current_project(db)
    if db_project is None:
        error_message = f"No project found with id {current_id}"
    else:
        db_project.name = rename
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            error_message = "Duplicate project name."

    db_project = get_current_project(db)
    projects = get_all_projects(db)

    return templates.TemplateResponse(
        "main_template.html",
        {
            "request": request,
            "project": db_project,
            "projects": projects,
            "error_message": error_message
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
def get_llms_options(
    project_id: int = Query(None),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    llms = db.query(LLM).offset(skip).limit(limit).all()

    selected_model = None
    if project_id:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            selected_model = project.llm_id  # or project.llm.id if llm is loaded

    llm_html = ""
    for llm in llms:
        selected_attr = " selected" if llm.id == selected_model else ""
        llm_html += f'<option value="{llm.id}"{selected_attr}>{llm.name}</option>'

    return llm_html



if __name__ == "__main__":
    insert_llms()
    import uvicorn
    #Note on the server this should run on 0.0.0.0
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
