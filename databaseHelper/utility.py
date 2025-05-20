from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

db_url = "sqlite:///./sqlite.db"

engine = create_engine(db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
       Dependency generator that provides a SQLAlchemy database session.

       This function is intended for use with FastAPI's dependency injection system.
       It creates a new database session using `SessionLocal`, yields it to the caller,
       and ensures the session is closed after use.

       This allows safe handling of database resources within request/response cycles
       and ensures that connections are properly cleaned up.

       Yields
       ------
       sqlalchemy.orm.Session
           A SQLAlchemy session object for interacting with the database.
       Notes
       -----
       The Name of the database can be configured via the db_url variable.
       """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
