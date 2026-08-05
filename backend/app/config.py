"""Application configuration.

All settings can be overridden via environment variables so the platform can be
deployed anywhere (Render, Railway, Docker, ...) without code changes.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads backend/.env; real env vars take precedence
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
for _d in (UPLOAD_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings:
    """Central settings object loaded from the environment."""

    APP_NAME: str = "ModelMind AI"
    APP_VERSION: str = "1.0.0"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'ml_platform.db'}"
    )

    # JWT configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

    # SMTP / email OTP verification
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    OTP_EXPIRE_MINUTES: int = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))
    OTP_MAX_ATTEMPTS: int = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))

    @property
    def SMTP_ENABLED(self) -> bool:
        """True when a real SMTP server is configured (real email delivery)."""
        return bool(self.SMTP_HOST and self.SMTP_PORT)

    # CORS
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")
        if o.strip()
    ]

    # ML pipeline defaults
    DEFAULT_TEST_SIZE: float = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
    DEFAULT_RANDOM_STATE: int = int(os.getenv("DEFAULT_RANDOM_STATE", "42"))

    # LLM labeling (Outcome column generation)
    # LLM_PROVIDER: "off" | "openai" | "ollama"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "off")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

    @property
    def LLM_ENABLED(self) -> bool:
        return self.LLM_PROVIDER.lower() in ("openai", "ollama")


settings = Settings()
