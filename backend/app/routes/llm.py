"""LLM-powered dataset labeling routes.

``POST /api/datasets/{dataset_id}/llm-label`` asks the configured LLM
(OpenAI-compatible or local Ollama) to design Outcome categories for a dataset
and label every row, then persists the new column as a versioned CSV.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Dataset, User
from ..schemas import LLMLabelRequest, LLMLabelResponse, LLMStatus
from ..services import llm as llm_svc
from ..services import preprocessing as pp
from ..services import storage
from ..services.pipeline import load_dataset_df
from ..utils.serialization import dumps, loads
from ..utils.security import get_current_user

router = APIRouter(prefix="/api", tags=["llm"])


@router.get("/llm/status", response_model=LLMStatus)
def llm_status():
    return llm_svc.llm_status()


@router.post("/datasets/{dataset_id}/llm-label", response_model=LLMLabelResponse)
def llm_label(
    dataset_id: int,
    payload: LLMLabelRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ds, df = load_dataset_df(db, dataset_id, user.id)

    if not settings.LLM_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="LLM is disabled. Set LLM_PROVIDER=openai or ollama in the environment / .env.",
        )

    column = payload.column_name.strip() or "Outcome"
    if column in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' already exists in this dataset.",
        )

    try:
        categories = llm_svc.propose_outcomes(df, payload.num_categories)
        names = [c["name"] for c in categories]
        labels = llm_svc.label_rows(
            df, names,
            batch_size=payload.batch_size,
            max_rows=payload.max_rows,
        )
    except llm_svc.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Attach the labels and persist as a new version of the dataset.
    out = df.copy()
    out[column] = labels[: len(out)]
    for c in (column,):
        out[c] = out[c].astype(object)

    import pandas as pd

    path = storage.save_upload(
        f"{ds.name.rsplit('.', 1)[0]}_llm_{column.lower()}.csv",
        out.to_csv(index=False).encode("utf-8"),
    )

    counts = out[column].value_counts(dropna=False).astype(int).to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    labeled_rows = int(sum(v for k, v in counts.items() if k != "Unknown"))

    profile = pp.profile_dataset(out)
    prev_versions = loads(ds.versions) if ds.versions else []
    prev_versions = prev_versions if isinstance(prev_versions, list) else []
    prev_versions.append({
        "version": len(prev_versions) + 1,
        "path": str(path),
        "rows": int(out.shape[0]),
        "note": f"LLM-generated '{column}' column",
    })

    ds.filepath = str(path)
    ds.rows = int(out.shape[0])
    ds.columns = dumps([str(c) for c in out.columns])
    ds.preview = dumps(out.head(10).fillna("").astype(str).to_dict(orient="records"))
    ds.profile = dumps(profile)
    ds.versions = dumps(prev_versions)
    db.add(ds)
    db.commit()
    db.refresh(ds)

    return LLMLabelResponse(
        dataset_id=dataset_id,
        column_name=column,
        categories=categories,
        labeled_rows=labeled_rows,
        counts=counts,
        preview=out.head(10).fillna("").astype(str).to_dict(orient="records"),
        filepath=str(path),
    )
