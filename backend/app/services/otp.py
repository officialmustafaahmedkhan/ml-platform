"""Email OTP verification using SMTP (stdlib ``smtplib``).

Design
------
* ``issue_otp`` generates a 6-digit code, stores only its SHA-256 hash plus an
  expiry timestamp on the user row, and dispatches the email via SMTP.
* When SMTP is not configured (``settings.SMTP_ENABLED == False``) the raw code
  is returned to the caller (dev mode) so the whole flow still works locally.
* ``verify_otp`` enforces expiry + a maximum number of attempts, then marks the
  account ``email_verified`` on success.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..config import settings
from ..models import User


def _now() -> datetime:
    """Naive-UTC now (matches how SQLite round-trips DateTime columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def generate_otp(length: int = 6) -> str:
    return f"{secrets.randbelow(10 ** length):0{length}d}"


def hash_otp(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def send_otp_email(to_email: str, code: str) -> bool:
    """Dispatch the OTP email over SMTP.

    Returns ``True`` when the message was handed to the SMTP server. Raises
    ``RuntimeError`` when SMTP is enabled but delivery fails.
    """
    if not settings.SMTP_ENABLED:
        return False

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = f"Your {settings.APP_NAME} verification code"
    msg["From"] = settings.SMTP_FROM or settings.SMTP_USERNAME
    msg["To"] = to_email
    msg.set_content(
        "Your verification code is: {code}\n\n"
        "It expires in {minutes} minutes.\n"
        "If you did not request this, you can safely ignore this email.".format(
            code=code, minutes=settings.OTP_EXPIRE_MINUTES
        )
    )
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"SMTP delivery failed: {exc}") from exc


def issue_otp(user: User, db: Session) -> str | None:
    """Generate, persist and email a fresh OTP.

    Returns the raw code **only** in dev mode (SMTP disabled) so local testing
    and the bundled test suite work without a mail server; otherwise ``None``.
    """
    code = generate_otp()
    user.otp_code = hash_otp(code)
    user.otp_expires_at = _now() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    user.otp_attempts = 0
    db.add(user)
    db.commit()

    dispatched = send_otp_email(user.email, code)
    return None if dispatched else code


def verify_otp(user: User, code: str, db: Session) -> bool:
    """Validate the submitted code; on success mark the account verified."""
    if not user.otp_code or not user.otp_expires_at:
        return False
    if user.otp_attempts >= settings.OTP_MAX_ATTEMPTS:
        return False
    if _now() > user.otp_expires_at:
        return False
    if not hmac.compare_digest(user.otp_code, hash_otp((code or "").strip())):
        user.otp_attempts += 1
        db.add(user)
        db.commit()
        return False

    user.email_verified = True
    user.otp_code = None
    user.otp_expires_at = None
    user.otp_attempts = 0
    db.add(user)
    db.commit()
    return True
