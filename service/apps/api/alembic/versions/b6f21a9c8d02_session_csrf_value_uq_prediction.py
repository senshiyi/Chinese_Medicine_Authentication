"""sessions.csrf_value + unique prediction per analysis job

Revision ID: b6f21a9c8d02
Revises: c7a468151dba
Create Date: 2026-09-10 00:00:00.000000

- sessions 增加 csrf_value（明文，仅服务端保存）：GET /me 回读当前 CSRF，
  不再每次轮换，避免前端并发 /me 造成 token 轮换竞态。
- predictions.analysis_job_id 增加唯一索引：数据库层保证一个分析 job
  至多一条预测（Worker 原子领取的硬兜底）。

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = 'b6f21a9c8d02'
down_revision = 'c7a468151dba'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column('csrf_value', sa.String(length=200), nullable=True))
    op.create_index('uq_predictions_analysis_job', 'predictions', ['analysis_job_id'], unique=True)


def downgrade() -> None:
    op.drop_index('uq_predictions_analysis_job', table_name='predictions')
    op.drop_column('sessions', 'csrf_value')
