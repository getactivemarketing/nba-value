"""Add home/away and L10 record columns to team_stats

Revision ID: 002
Revises: 001
Create Date: 2026-01-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to team_stats table
    op.add_column('team_stats', sa.Column('wins_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('losses_l10', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('home_wins', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('home_losses', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('away_wins', sa.Integer(), nullable=True))
    op.add_column('team_stats', sa.Column('away_losses', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('team_stats', 'away_losses')
    op.drop_column('team_stats', 'away_wins')
    op.drop_column('team_stats', 'home_losses')
    op.drop_column('team_stats', 'home_wins')
    op.drop_column('team_stats', 'losses_l10')
    op.drop_column('team_stats', 'wins_l10')
