"""Add ATS and O/U tracking columns to team_stats

Revision ID: 003
Revises: 002
Create Date: 2026-01-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add ATS (Against the Spread) tracking columns
    op.add_column('team_stats', sa.Column('ats_wins_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('ats_losses_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('ats_pushes_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('ats_pct_l10', sa.Numeric(4, 3), nullable=True))

    # Add O/U (Over/Under) tracking columns
    op.add_column('team_stats', sa.Column('ou_overs_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('ou_unders_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('ou_pushes_l10', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('team_stats', 'ou_pushes_l10')
    op.drop_column('team_stats', 'ou_unders_l10')
    op.drop_column('team_stats', 'ou_overs_l10')
    op.drop_column('team_stats', 'ats_pct_l10')
    op.drop_column('team_stats', 'ats_pushes_l10')
    op.drop_column('team_stats', 'ats_losses_l10')
    op.drop_column('team_stats', 'ats_wins_l10')
