"""Natural-language ML command routes.

``POST /api/commands`` accepts plain-English commands (see
``services.commands``) and returns a structured, human-readable result.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..services import commands as cmd
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/commands", tags=["commands"])


class CommandRequest(BaseModel):
    text: str


@router.post("")
def run_command(payload: CommandRequest, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Command text is required")
    parsed = cmd.parse_command(payload.text)
    try:
        return cmd.execute_command(parsed, db, user)
    except ValueError as exc:
        return {
            "action": parsed.get("action", "unknown"),
            "summary": str(exc),
            "cards": [], "rows": None, "text": None, "detail": None,
        }


@router.get("/help")
def command_help(user: User = Depends(get_current_user)):
    return {
        "commands": [
            {"command": k, "description": v}
            for k, v in cmd._COMMANDS.items()
        ]
    }
