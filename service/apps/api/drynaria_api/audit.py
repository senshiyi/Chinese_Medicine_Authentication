"""审计日志辅助：安全与业务关键事件写入 audit_logs。"""

from __future__ import annotations

from uuid import UUID

from fastapi import Request
from sqlalchemy.orm import Session

from .models import AuditLog


def record_audit(
    db: Session,
    *,
    action: str,
    actor_user_id: UUID | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict | None = None,
    request: Request | None = None,
) -> AuditLog:
    ip_address = None
    request_id = None
    if request is not None:
        ip_address = request.client.host if request.client else None
        request_id = getattr(request.state, "request_id", None)
    row = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        detail=detail,
        ip_address=ip_address,
        request_id=request_id,
    )
    db.add(row)
    return row
