"""predictions.scope_decision

Revision ID: c9d4e2f1a8b0
Revises: b6f21a9c8d02
Create Date: 2026-09-10 00:00:00.000000

- predictions 增加 scope_decision（JSON，可空）：记录该预测产生时的范围门控决策
  （accepted / rejected / unavailable + reason / version / score / threshold 等）。
  None 表示该预测产生于未启用范围门控的阶段（历史预测），或该次推理未记录范围决策。

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = 'c9d4e2f1a8b0'
down_revision = 'b6f21a9c8d02'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('predictions', sa.Column('scope_decision', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('predictions', 'scope_decision')
