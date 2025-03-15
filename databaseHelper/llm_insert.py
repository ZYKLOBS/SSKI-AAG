from sqlalchemy.orm import Session
from models.llm import LLM
from databaseHelper.utility import SessionLocal

llm_names = ["ollama", "claude"]

def insert_llms():
    db = SessionLocal()
    try:
        for name in llm_names:
            if not db.query(LLM).filter_by(name=name).first():
                db.add(LLM(name=name))
        db.commit()
    except Exception as e:
        print(f"Error inserting LLMs: {e}")
        db.rollback()
    finally:
        db.close()
