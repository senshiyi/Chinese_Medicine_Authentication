"""SQLAlchemy 2 engine / session 工厂与应用内依赖。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def make_engine(database_url: str) -> Engine:
    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    if database_url.startswith("sqlite"):
        # 测试常用 SQLite；逐连接开启外键约束，贴近 PostgreSQL 行为。
        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = make_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def create_all(database_url: str) -> None:
    """测试/本地直连建表；生产环境用 Alembic 迁移。"""
    engine = make_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()


def get_db(request: Request) -> Iterator[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]
