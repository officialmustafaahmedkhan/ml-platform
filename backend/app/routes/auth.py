"""Authentication & user management routes (email OTP verification).

Flow
----
1. ``POST /register``      -> creates an unverified account, emails an OTP.
2. ``POST /verify-otp``    -> checks the code, marks the email verified and
                              returns a JWT (auto-login).
3. ``POST /send-otp``      -> resend the code (also triggered on login when
                              the account is not verified yet).
4. ``POST /login``         -> only succeeds for verified accounts.

When SMTP is not configured the API runs in *dev mode*: the OTP is returned in
the response body so the flow works locally without a mail server.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import User
from ..schemas import (
    AuthResponse,
    LoginRequest,
    RegisterOut,
    RegisterRequest,
    SendOtpOut,
    SendOtpRequest,
    UserOut,
    VerifyOtpRequest,
)
from ..services import otp as otp_svc
from ..utils.serialization import dumps
from ..utils.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_response(user: User) -> AuthResponse:
    return AuthResponse(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@router.post("/register", response_model=RegisterOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=payload.email.lower(),
        name=payload.name.strip(),
        password_hash=hash_password(payload.password),
        preferences=dumps({}),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    dev_otp = otp_svc.issue_otp(user, db)
    if dev_otp is not None:
        return RegisterOut(
            email=user.email,
            needs_verification=True,
            dev_otp=dev_otp,
            message="Account created. Enter the OTP below. (Dev mode: no SMTP configured — the code is shown here.)",
        )
    return RegisterOut(
        email=user.email,
        needs_verification=True,
        dev_otp=None,
        message=f"Account created. A verification code was emailed to {user.email}.",
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.email_verified:
        dev_otp = otp_svc.issue_otp(user, db)
        raise HTTPException(
            status_code=403,
            detail={
                "code": "email_not_verified",
                "message": "Your email is not verified yet. Enter the code sent to your inbox.",
                "email": user.email,
                "dev_otp": dev_otp,
            },
        )
    return _auth_response(user)


@router.post("/send-otp", response_model=SendOtpOut)
def send_otp(payload: SendOtpRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None:
        raise HTTPException(status_code=404, detail="No account found for that email")
    if user.email_verified:
        return SendOtpOut(
            email=user.email, dev_otp=None,
            message="Email already verified.",
            expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
        )
    dev_otp = otp_svc.issue_otp(user, db)
    return SendOtpOut(
        email=user.email, dev_otp=dev_otp,
        message="Verification code sent.",
        expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
    )


@router.post("/verify-otp", response_model=AuthResponse)
def verify_otp(payload: VerifyOtpRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None:
        raise HTTPException(status_code=404, detail="No account found for that email")
    if user.email_verified:
        return _auth_response(user)
    if not otp_svc.verify_otp(user, payload.otp, db):
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")
    return _auth_response(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.model_validate(user)
