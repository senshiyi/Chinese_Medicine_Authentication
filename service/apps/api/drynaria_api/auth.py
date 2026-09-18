"""鉴权路由：注册/登录/登出/当前用户；管理员创建用户（user/expert/admin）。

CSRF 语义：token 在登录/注册签发新会话时生成并绑定；GET /me 只读回读当前
CSRF（不轮换），因此前端任意次数调用 /me 都不会让客户端 token 与服务端失配
（登录后页面挂载 + 路由守卫各调一次 /me 是安全的）。
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response, status as http_status
from sqlalchemy.orm import Session

from .audit import record_audit
from .database import DbSession
from .models import Session as AuthSession
from .models import User, UserRole
from .schemas import (
    AdminCreateUserRequest,
    LoginRequest,
    MeOut,
    RegisterRequest,
    UserOut,
)
from .security import (
    CsrfGuard,
    CurrentUser,
    AuthError,
    _sha256_hex,
    hash_password,
    issue_session,
    lookup_user_by_email,
    me_out,
    new_user,
    resolve_session,
    session_cookie_params,
    verify_password,
)
from .settings import Settings, get_app_settings, get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _set_session_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(**session_cookie_params(settings), value=token)


def _clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(settings.session_cookie, path="/")


def _csrf_for_session(db: Session, raw_token: str | None) -> str | None:
    """返回当前会话绑定的 CSRF 明文，供 GET /me 回读（只读，不轮换）。

    旧数据兜底：仅当该会话创建于 csrf_value 列引入之前（值为空）时，轮换
    一次并持久化明文；此后 /me 恒定返回同一 token。
    """
    session = resolve_session(db, raw_token)
    if session is None:
        return None
    if session.csrf_value:
        return session.csrf_value
    fresh = secrets.token_urlsafe(32)
    session.csrf_hash = _sha256_hex(fresh)
    session.csrf_value = fresh
    db.commit()
    return fresh


@router.post("/register", status_code=http_status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    response: Response,
    db: DbSession,
    settings: Settings = Depends(get_app_settings),
) -> MeOut:
    if lookup_user_by_email(db, body.email) is not None:
        raise AuthError("email_taken", "邮箱已被注册。", http=http_status.HTTP_409_CONFLICT)
    user = new_user(db, email=body.email, password=body.password, role=UserRole.User, display_name=body.display_name)
    session_token, csrf_token = issue_session(db, user, settings)
    record_audit(db, action="auth.register", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
    db.commit()
    _set_session_cookie(response, settings, session_token)
    return me_out(user, csrf_token)


@router.post("/login")
def login(body: LoginRequest, response: Response, db: DbSession, settings: Settings = Depends(get_app_settings)) -> MeOut:
    user = lookup_user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise AuthError("invalid_credentials", "邮箱或密码错误。")
    if not user.is_active:
        raise AuthError("account_disabled", "账户不可用。")
    session_token, csrf_token = issue_session(db, user, settings)
    record_audit(db, action="auth.login", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
    db.commit()
    _set_session_cookie(response, settings, session_token)
    return me_out(user, csrf_token)


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
    settings: Settings = Depends(get_app_settings),
) -> dict:
    from .security import _sha256_hex, resolve_session  # noqa: PLC0415

    token = request.cookies.get(settings.session_cookie)
    session = resolve_session(db, token)
    if session is not None and session.user_id == user.id:
        session.revoked_at = datetime.now(timezone.utc)
    record_audit(
        db, action="auth.logout", actor_user_id=user.id, resource_type="user", resource_id=str(user.id)
    )
    db.commit()
    _clear_session_cookie(response, settings)
    return {"ok": True}


@router.get("/me")
def me(request: Request, db: DbSession, user: CurrentUser) -> MeOut:
    settings = get_app_settings(request)
    csrf = _csrf_for_session(db, request.cookies.get(settings.session_cookie))
    return me_out(user, csrf or "")


@admin_router.post("/users", status_code=http_status.HTTP_201_CREATED)
def admin_create_user(
    body: AdminCreateUserRequest,
    db: DbSession,
    admin: CurrentUser,
    _csrf: CsrfGuard,
) -> UserOut:
    if lookup_user_by_email(db, body.email) is not None:
        raise AuthError("email_taken", "邮箱已被注册。", http=http_status.HTTP_409_CONFLICT)
    role = UserRole(body.role)
    if admin.role != UserRole.Admin:
        raise AuthError("forbidden", "仅管理员可创建账户。", http=http_status.HTTP_403_FORBIDDEN)
    user = new_user(db, email=body.email, password=body.password, role=role, display_name=body.display_name)
    record_audit(
        db,
        action="admin.user.create",
        actor_user_id=admin.id,
        resource_type="user",
        resource_id=str(user.id),
        detail={"role": body.role},
    )
    db.commit()
    return UserOut(id=user.id, email=user.email, role=role.value, display_name=user.display_name)
