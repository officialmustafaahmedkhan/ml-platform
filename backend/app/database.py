"""SQLAlchemy database engine, session and declarative base.

Uses SQLite by default (zero-config). Set ``DATABASE_URL`` to point at
Postgres/MySQL/Mongo-compatible SQL layer in production.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings


def _normalize_url(url: str) -> str:
    """Map plain ``postgresql://`` URLs onto the psycopg3 driver.

    Serverless hosts (Vercel, Neon, ...) hand out ``postgresql://`` URLs;
    SQLAlchemy needs an explicit driver, so we default to ``+psycopg``.
    """
    if url.startswith("sqlite"):
        return url
    scheme, _, rest = url.partition("://")
    if scheme in ("postgres", "postgresql"):
        return f"postgresql+psycopg://{rest}"
    return url


_DATABASE_URL = _normalize_url(settings.DATABASE_URL)
_IS_SQLITE = _DATABASE_URL.startswith("sqlite")

# SQLite is single-writer: give it a busy timeout so concurrent writers wait
# instead of failing with "database is locked" (default timeout is 0ms).
if _IS_SQLITE:
    connect_args = {"check_same_thread": False, "timeout": 30}
    engine_kwargs = {}
else:
    # Serverless/stateless engines must not hold pooled connections across
    # invocations; open a fresh connection per call and ping before use.
    from sqlalchemy.pool import NullPool

    connect_args = {"connect_timeout": 10}
    engine_kwargs = {"poolclass": NullPool, "pool_pre_ping": True}

engine = create_engine(
    _DATABASE_URL, connect_args=connect_args, future=True, **engine_kwargs
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


if _IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

Base = declarative_base()


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Import models so metadata is populated."""
    from . import models  # noqa: F401  (registers tables on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _ensure_user_columns()
    _ensure_experiment_columns()


def _ensure_user_columns() -> None:
    """Lightweight SQLite migration: add newly-introduced User columns to an
    existing database without requiring the user to delete it."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return
    from .models import User

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(users)"))}
        added = []
        for col in User.__table__.columns:
            if col.name not in existing:
                coltype = col.type.compile(engine.dialect)
                default = ""
                if col.default is not None and col.default.arg is not None:
                    arg = col.default.arg
                    if isinstance(arg, bool):
                        default = f" DEFAULT {1 if arg else 0}"
                    elif isinstance(arg, (int, float)):
                        default = f" DEFAULT {arg}"
                    elif isinstance(arg, str):
                        default = f" DEFAULT '{arg}'"
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col.name} {coltype}{default}"))
                added.append(col.name)
        conn.commit()
        if added:
            print(f"[init_db] added missing user columns: {added}")


def _ensure_experiment_columns() -> None:
    """Lightweight migration: add the `notes` column to existing experiments tables."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return
    from .models import Experiment

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(experiments)"))}
        added = []
        for col in Experiment.__table__.columns:
            if col.name not in existing:
                coltype = col.type.compile(engine.dialect)
                conn.execute(text(f"ALTER TABLE experiments ADD COLUMN {col.name} {coltype} DEFAULT ''"))
                added.append(col.name)
        conn.commit()
        if added:
            print(f"[init_db] added missing experiment columns: {added}")
