"""Export routes: download pickled models + generated CSV reports."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from ..config import REPORT_DIR
from ..database import get_db
from ..models import ModelArtifact, User
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/model/{model_id}")
def export_model(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        m.filepath,
        media_type="application/octet-stream",
        filename=f"{m.model_type}_model_{m.id}.pkl",
    )


@router.get("/report/{model_id}")
def export_report(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download a CSV report with the model's evaluation metrics."""
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")

    stored = json.loads(m.metrics) if m.metrics else {}
    metrics = stored.get("metrics", {})
    rows = [
        {"metric": k, "value": v} for k, v in metrics.items()
        if isinstance(v, (int, float)) and k not in ("samples",)
    ]
    for pc in metrics.get("per_class", []):
        rows.append({"metric": f"class_{pc['class']}_precision", "value": pc["precision"]})
        rows.append({"metric": f"class_{pc['class']}_recall", "value": pc["recall"]})
        rows.append({"metric": f"class_{pc['class']}_f1", "value": pc["f1"]})

    import io

    import pandas as pd

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)

    from fastapi.responses import Response

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={m.model_type}_report_{m.id}.csv"},
    )


@router.get("/report/{model_id}/pdf")
def export_report_pdf(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download an automatically generated PDF report for a model."""
    from ..services import report as report_svc

    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        pdf = report_svc.build_model_report(m, db)
    except Exception as exc:  # noqa: BLE001 - surface a clean error
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={m.model_type}_report_{m.id}.pdf"},
    )


@router.get("/batch/{filename}")
def download_batch(filename: str, user: User = Depends(get_current_user)):
    """Download a previously generated batch-prediction CSV report."""
    path = REPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/csv", filename=filename)
