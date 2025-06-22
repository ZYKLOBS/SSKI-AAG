from models.llm import LLM
from databaseHelper.utility import SessionLocal

#llm_names = ["ollama", "claude"]

llm_names = ["claude", "ollama"]
def insert_llms() -> None:
    """
    Insert predefined LLM names into the database if they do not already exist.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    Exception
        If an error occurs during the database transaction. The error is caught and printed.

    Notes
    -----
    The list of LLM names to insert is defined in the `llm_names` variable. Modify this list
    to include additional LLM names as needed.
    """
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
