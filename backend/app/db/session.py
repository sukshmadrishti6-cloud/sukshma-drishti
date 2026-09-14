"""Database Engine and Session Management Module for SukshmaDrishti."""
import logging
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure engine with SQLite thread check bypass if using SQLite
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


_db_initialized = False


def init_db() -> None:
    """Initializes database schema tables."""
    global _db_initialized
    try:
        from app.db import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        _db_initialized = True
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e!s}", exc_info=True)
        raise


def get_db() -> Generator[Session, None, None]:
    """Dependency yielding transactional SQLAlchemy database session."""
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

