"""Pydantic v2 schemas used across the API surface."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# --------------------------------------------------------------------------- #
# Auth / users
# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    name: str
    preferences: Optional[dict] = None
    email_verified: bool = False
    created_at: datetime

    @field_validator("preferences", mode="before")
    @classmethod
    def _parse_preferences(cls, v):
        import json

        if isinstance(v, str) and v:
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return None
        return v


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class RegisterOut(BaseModel):
    """Returned by ``/register`` — no token until the email is verified."""
    email: EmailStr
    needs_verification: bool = True
    dev_otp: Optional[str] = None  # dev mode only (SMTP not configured)
    message: str


class SendOtpRequest(BaseModel):
    email: EmailStr


class SendOtpOut(BaseModel):
    email: EmailStr
    dev_otp: Optional[str] = None
    message: str
    expires_in_minutes: int


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=4, max_length=8)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class DatasetOut(BaseModel):
    id: int
    name: str
    filename: str
    rows: int
    columns: Optional[list[str]] = None
    preview: Optional[list[dict]] = None
    profile: Optional[dict] = None
    versions: Optional[list] = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# LLM labeling
# --------------------------------------------------------------------------- #
class LLMStatus(BaseModel):
    enabled: bool
    provider: str
    model: Optional[str] = None
    api_key_configured: bool = False
    endpoint: Optional[str] = None


class LLMLabelRequest(BaseModel):
    column_name: str = "Outcome"
    num_categories: int = Field(default=3, ge=2, le=6)
    batch_size: int = Field(default=25, ge=1, le=100)
    max_rows: Optional[int] = Field(default=None, ge=1)


class LLMLabelResponse(BaseModel):
    dataset_id: int
    column_name: str
    categories: list[dict]
    labeled_rows: int
    counts: dict
    preview: list[dict]
    filepath: str


class AssistantMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    dataset_id: Optional[int] = None
    model_id: Optional[int] = None
    history: list[AssistantMessage] = []


# --------------------------------------------------------------------------- #
# Experiment workspace (history + notes)
# --------------------------------------------------------------------------- #
class ExperimentOut(BaseModel):
    id: int
    action: str
    details: dict = {}
    dataset_id: Optional[int] = None
    model_id: Optional[int] = None
    notes: str = ""
    created_at: datetime
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None


class ExperimentUpdate(BaseModel):
    notes: str = Field(default="", max_length=2000)


class AssistantResponse(BaseModel):
    reply: str
    context: dict = {}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class PreprocessRequest(BaseModel):
    """Request body for the preprocessing endpoint.

    ``mode="auto"`` uses the recommendation-driven pipeline; ``mode="manual"``
    takes the explicit strategy overrides below.
    """

    dataset_id: int
    mode: str = "auto"  # "auto" | "manual"
    target_column: Optional[str] = None
    missing_numeric: str = "mean"  # mean | median | mode | drop
    missing_categorical: str = "mode"  # mode | drop | constant
    encoding: str = "auto"  # auto | label | onehot
    scaling: str = "auto"  # auto | standard | minmax | none
    smote: bool = False
    drop_columns: Optional[list[str]] = None


class PreprocessResponse(BaseModel):
    dataset_id: int
    mode: str
    profile: dict
    config_used: dict
    report: dict
    recommendation: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class TrainRequest(BaseModel):
    dataset_id: int
    model_type: str  # dt | knn | rf | voting | stacking
    target_column: Optional[str] = None
    params: dict = Field(default_factory=dict)
    preprocess: dict = Field(default_factory=dict)  # any subset of PreprocessRequest fields
    test_size: float = 0.2
    random_state: int = 42
    tune: bool = False  # run GridSearchCV-lite on base models
    cv_folds: int = 5  # used by GridSearchCV lite


class TrainResponse(BaseModel):
    model_id: int
    model_type: str
    name: str
    params: dict
    metrics: dict
    feature_names: list[str]
    class_names: list[str]
    dataset_id: int


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    model_type: str
    params: dict
    pipeline: dict
    metrics: dict
    feature_names: list[str]
    class_names: list[str]
    dataset_id: int
    created_at: datetime


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class EvaluateResponse(BaseModel):
    model_id: int
    model_type: str
    metrics: dict
    confusion_matrix: list[list[int]]
    class_names: list[str]
    feature_importance: Optional[dict] = None
    charts: dict  # base64 data-uri encoded PNGs


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
class PredictResponse(BaseModel):
    model_id: int
    prediction: Any
    probabilities: Optional[dict] = None
    explanation: Optional[list[dict]] = None


class BatchPredictResponse(BaseModel):
    model_id: int
    total: int
    results: list[dict]
    output_filename: str


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
class ComparisonRequest(BaseModel):
    dataset_id: int
    target_column: Optional[str] = None
    preprocess: dict = Field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    include_hybrid: bool = True
    model_types: Optional[list[str]] = None  # explicit algorithm selection (overrides include_hybrid)


class ComparisonResponse(BaseModel):
    table: list[dict]
    charts: dict
    best_model: dict
    dataset_id: int


# --------------------------------------------------------------------------- #
# Recommendation engine
# --------------------------------------------------------------------------- #
class RecommendationResponse(BaseModel):
    dataset_id: int
    dataset_profile: dict
    model_recommendations: list[dict]
    preprocessing_recommendations: list[dict]
    improvement_suggestions: list[dict]
    predicted_best_model: dict


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class DashboardResponse(BaseModel):
    user: UserOut
    stats: dict
    recent_datasets: list[dict]
    recent_models: list[dict]
    accuracy_trend: list[dict]
    activity_timeline: list[dict]
    model_type_distribution: dict
    suggestions: list[dict]
