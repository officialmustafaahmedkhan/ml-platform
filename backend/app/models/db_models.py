"""Database schema for users, datasets, trained models and experiment history.

Schema overview
---------------
User
  - id, email, name, password_hash, preferences (JSON), created_at
  - datasets[]   -> saved datasets
  - models[]     -> trained model artifacts
  - experiments[] -> activity/history timeline

Dataset
  - metadata + stored CSV path + a lightweight `profile` (shape, dtypes,
    missingness, imbalance) used by the recommendation engine
  - `versions` keeps track of dataset versioning (initial upload + re-uploads)

ModelArtifact
  - a persisted trained model (pickle file) plus the evaluation `metrics`,
    the `feature_names`, and the preprocessing `pipeline` used to build it so
    predictions can be reconstructed later.

Experiment
  - append-only log of every action a user performs (upload, preprocess,
    train, predict, ...) -> powers the dashboard timeline + history page.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    preferences: Mapped[str] = mapped_column(Text, default="{}")  # JSON {theme, ...}
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    otp_code: Mapped[str | None] = mapped_column(String(128), nullable=True)  # SHA-256 of the OTP
    otp_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    otp_attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    models: Mapped[list["ModelArtifact"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    columns: Mapped[str] = mapped_column(Text, default="[]")  # JSON list[str]
    preview: Mapped[str] = mapped_column(Text, default="[]")  # JSON head rows
    profile: Mapped[str] = mapped_column(Text, default="{}")  # JSON profile dict
    versions: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of {version, path}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="datasets")
    models: Mapped[list["ModelArtifact"]] = relationship(back_populates="dataset")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="dataset")


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # dt/knn/rf/voting/stacking
    params: Mapped[str] = mapped_column(Text, default="{}")  # JSON hyperparameters
    pipeline: Mapped[str] = mapped_column(Text, default="{}")  # JSON preprocessing config
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[str] = mapped_column(Text, default="{}")  # JSON evaluation metrics
    feature_names: Mapped[str] = mapped_column(Text, default="[]")  # JSON list[str]
    class_names: Mapped[str] = mapped_column(Text, default="[]")  # JSON list[str]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="models")
    dataset: Mapped["Dataset"] = relationship(back_populates="models")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="model")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)  # upload/preprocess/train/predict/compare
    details: Mapped[str] = mapped_column(Text, default="{}")  # JSON payload summary
    notes: Mapped[str] = mapped_column(Text, default="")  # user annotation for the experiment
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="experiments")
    dataset: Mapped["Dataset"] = relationship(back_populates="experiments")
    model: Mapped["ModelArtifact"] = relationship(back_populates="experiments")
