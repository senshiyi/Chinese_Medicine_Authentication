"""本地账户安全：Argon2id 密码哈希、服务端不透明会话、CSRF、角色依赖。"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import Session as AuthSession
from .models import User, UserRole
from .schemas import MeOut, UserOut
from .settings import Settings, get_settings

_password_hasher = PasswordHasher()

SESSION_COOKIE = "drynaria_session"
CSRF_HEADER = "X-CSRF-Token"


class AuthError(HTTPException):
    def __init__(self, code: str, message: str, http: int = status.HTTP_401_UNAUTHORIZED) -> None:
        super().__init__(
            status_code=http, detail={"code": code, "message": message}
        )


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError, Argon2Error):
        return False


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_session(db: Session, user: User, settings: Settings) -> tuple[str, str]:
    """创建服务端会话；返回 (session_token, csrf_token)。数据库只存 token 哈希。

    csrf 明文也保存在 sessions.csrf_value（服务端库内），供 GET /me 只读回读；
    /me 不轮换 CSRF，从根上消除多次 /me 的轮换竞态。
    """
    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.session_ttl_days)
    row = AuthSession(
        user_id=user.id,
        token_hash=_sha256_hex(session_token),
        csrf_hash=_sha256_hex(csrf_token),
        csrf_value=csrf_token,
        expires_at=expires_at,
    )
    db.add(row)
    return session_token, csrf_token


def resolve_session(db: Session, raw_token: str | None) -> AuthSession | None:
    if not raw_token:
        return None
    row = db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == _sha256_hex(raw_token),
            AuthSession.revoked_at.is_(None),
        )
    )
    if row is None:
        return None
    # SQLite 无时区列：读回的是 naive 时间，统一按 UTC 语义比较
    expires_at = row.expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        return None
    return row


def _request_settings(request: Request) -> Settings:
    app_settings = getattr(request.app.state, "settings", None)
    return app_settings if isinstance(app_settings, Settings) else get_settings()


def current_user_dep(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    settings = _request_settings(request)
    raw = request.cookies.get(settings.session_cookie)
    session = resolve_session(db, raw)
    if session is None:
        raise AuthError("auth_required", "请先登录。")
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise AuthError("auth_required", "账户不可用。")
    return user


CurrentUser = Annotated[User, Depends(current_user_dep)]


def require_csrf(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """状态变更接口必须携带与当前会话绑定的 CSRF token（位于请求头）。"""
    settings = _request_settings(request)
    raw = request.cookies.get(settings.session_cookie)
    session = resolve_session(db, raw)
    if session is None:
        raise AuthError("auth_required", "请先登录。")
    header_value = request.headers.get(settings.csrf_header, "")
    expected = session.csrf_hash
    provided = _sha256_hex(header_value) if header_value else ""
    if not expected or not secrets.compare_digest(expected, provided):
        raise AuthError(
            "csrf_invalid", "缺少或错误的 CSRF token。", http=status.HTTP_403_FORBIDDEN
        )


CsrfGuard = Annotated[None, Depends(require_csrf)]


def require_roles(*roles: UserRole):
    def _checker(user: CurrentUser) -> User:
        if user.role not in roles:
            raise AuthError(
                "forbidden", "没有权限执行此操作。", http=status.HTTP_403_FORBIDDEN
            )
        return user

    return _checker


def user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        role=user.role.value if isinstance(user.role, UserRole) else user.role,
        display_name=user.display_name,
    )


def me_out(user: User, csrf_token: str) -> MeOut:
    return MeOut(user=user_out(user), csrf_token=csrf_token)


def session_cookie_params(settings: Settings) -> dict:
    return {
        "key": settings.session_cookie,
        "httponly": True,
        "secure": settings.cookie_secure_effective,
        "samesite": "lax",
        "path": "/",
    }


def lookup_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.strip().lower()))


def new_user(db: Session, *, email: str, password: str, role: UserRole, display_name: str) -> User:
    user = User(
        email=email.strip().lower(),
        password_hash=hash_password(password),
        role=role,
        display_name=display_name.strip(),
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user
