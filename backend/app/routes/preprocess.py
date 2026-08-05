"""Preprocessing routes: profile + auto/manual preprocessing with report."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import PreprocessRequest, PreprocessResponse
from ..services import preprocessing as pp
from ..services import recommendation as rec
from ..services.pipeline import prepare
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/preprocess", tags=["preprocessing"])


@router.get("/profile/{dataset_id}")
def dataset_profile(dataset_id: int, target: str | None = None,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from ..services.pipeline import load_dataset_df

    ds, df = load_dataset_df(db, dataset_id, user.id)
    return {"profile": pp.profile_dataset(df, target), "columns": [str(c) for c in df.columns]}


@router.post("/auto-config/{dataset_id}")
def get_auto_config(dataset_id: int, model_hint: str | None = None,
                    target: str | None = None,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the recommended config the engine *would* use (no processing)."""
    from ..services.pipeline import load_dataset_df, resolve_target

    ds, df = load_dataset_df(db, dataset_id, user.id)
    target_col = resolve_target(df, target)
    config = pp.auto_config(df, target_col, model_hint)
    return {"config": config, "target_column": target_col}


@router.post("", response_model=PreprocessResponse)
def run_preprocessing(payload: PreprocessRequest,
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        ds, df, target, X, y, pipeline, report, config, profile = prepare(
            db, payload.dataset_id, user.id,
            preprocess=payload.model_dump(exclude={"dataset_id"}),
            target_column=payload.target_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rec_all = rec.recommend_all(df, target)

    from ..routes.helpers import log_experiment

    log_experiment(db, user, "preprocess", dataset_id=payload.dataset_id,
                   details={"target": target, "config": config, "report": report})
    db.commit()

    return PreprocessResponse(
        dataset_id=payload.dataset_id,
        mode=payload.mode,
        profile=profile,
        config_used=config,
        report=report,
        recommendation=rec_all,
    )
