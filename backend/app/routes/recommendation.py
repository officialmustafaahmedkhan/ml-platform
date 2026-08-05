"""Recommendation engine routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import RecommendationResponse
from ..services import recommendation as rec
from ..services.pipeline import load_dataset_df, resolve_target
from ..utils.serialization import to_jsonable
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/recommend", tags=["recommendation"])


@router.get("/{dataset_id}", response_model=RecommendationResponse)
def recommend(dataset_id: int, target: str | None = None, model_hint: str | None = None,
              user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds, df = load_dataset_df(db, dataset_id, user.id)
    target_col = resolve_target(df, target)
    result = rec.recommend_all(df, target_col, model_hint=model_hint)
    result["dataset_id"] = dataset_id
    result["dataset_profile"]["target_column"] = target_col
    return RecommendationResponse(**to_jsonable(result))
