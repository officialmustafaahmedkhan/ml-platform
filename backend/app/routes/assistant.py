"""AI Assistant routes — chat about datasets and models."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, ModelArtifact, User
from ..schemas import AssistantRequest, AssistantResponse
from ..services import assistant as assistant_svc
from ..services import llm as llm_svc
from ..services.pipeline import load_dataset_df
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("/chat", response_model=AssistantResponse)
def chat(
    payload: AssistantRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dataset_context = None
    model_context = None

    if payload.dataset_id is not None:
        try:
            ds, df = load_dataset_df(db, payload.dataset_id, user.id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"Dataset not found: {exc}") from exc
        dataset_context = assistant_svc.build_dataset_context(ds, df)

    if payload.model_id is not None:
        m = (
            db.query(ModelArtifact)
            .filter(ModelArtifact.id == payload.model_id, ModelArtifact.user_id == user.id)
            .first()
        )
        if m is None:
            raise HTTPException(status_code=404, detail="Model not found")
        model_context = assistant_svc.build_model_context(m)

    if dataset_context is None and model_context is None:
        raise HTTPException(
            status_code=400,
            detail="Provide dataset_id and/or model_id so the assistant has something to talk about.",
        )

    history = [{"role": t.role, "content": t.content} for t in payload.history]

    try:
        result = assistant_svc.ask(
            payload.message,
            dataset_context=dataset_context,
            model_context=model_context,
            history=history,
        )
    except llm_svc.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AssistantResponse(reply=result["reply"], context=result["context"])
