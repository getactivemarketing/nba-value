"""Initial schema with all tables.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Teams table
    op.create_table(
        'teams',
        sa.Column('team_id', sa.String(10), primary_key=True),
        sa.Column('external_id', sa.Integer(), nullable=True),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('abbreviation', sa.String(10), nullable=False),
        sa.Column('city', sa.String(50), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('conference', sa.String(10), nullable=False),
        sa.Column('division', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Team stats table (rolling statistics)
    op.create_table(
        'team_stats',
        sa.Column('team_id', sa.String(10), primary_key=True),
        sa.Column('stat_date', sa.Date(), primary_key=True),
        sa.Column('games_played', sa.Integer(), nullable=False, server_default='0'),
        # Offensive Rating - rolling windows
        sa.Column('ortg_5', sa.Numeric(6, 2), nullable=True),
        sa.Column('ortg_10', sa.Numeric(6, 2), nullable=True),
        sa.Column('ortg_20', sa.Numeric(6, 2), nullable=True),
        sa.Column('ortg_season', sa.Numeric(6, 2), nullable=True),
        # Defensive Rating - rolling windows
        sa.Column('drtg_5', sa.Numeric(6, 2), nullable=True),
        sa.Column('drtg_10', sa.Numeric(6, 2), nullable=True),
        sa.Column('drtg_20', sa.Numeric(6, 2), nullable=True),
        sa.Column('drtg_season', sa.Numeric(6, 2), nullable=True),
        # Net Rating
        sa.Column('net_rtg_5', sa.Numeric(6, 2), nullable=True),
        sa.Column('net_rtg_10', sa.Numeric(6, 2), nullable=True),
        sa.Column('net_rtg_season', sa.Numeric(6, 2), nullable=True),
        # Pace
        sa.Column('pace_10', sa.Numeric(5, 2), nullable=True),
        sa.Column('pace_season', sa.Numeric(5, 2), nullable=True),
        # Points per game
        sa.Column('ppg_10', sa.Numeric(5, 2), nullable=True),
        sa.Column('ppg_season', sa.Numeric(5, 2), nullable=True),
        sa.Column('opp_ppg_10', sa.Numeric(5, 2), nullable=True),
        sa.Column('opp_ppg_season', sa.Numeric(5, 2), nullable=True),
        # Record
        sa.Column('wins', sa.Integer(), server_default='0'),
        sa.Column('losses', sa.Integer(), server_default='0'),
        sa.Column('win_pct_10', sa.Numeric(4, 3), nullable=True),
        # Rest context
        sa.Column('days_rest', sa.Integer(), nullable=True),
        sa.Column('is_back_to_back', sa.Boolean(), nullable=True),
        sa.Column('games_last_7_days', sa.Integer(), nullable=True),
        # Home/Away splits
        sa.Column('home_win_pct', sa.Numeric(4, 3), nullable=True),
        sa.Column('away_win_pct', sa.Numeric(4, 3), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_team_stats_team', 'team_stats', ['team_id'])
    op.create_index('idx_team_stats_date', 'team_stats', ['stat_date'])

    # Games table
    op.create_table(
        'games',
        sa.Column('game_id', sa.String(50), primary_key=True),
        sa.Column('league', sa.String(10), server_default='NBA'),
        sa.Column('season', sa.Integer(), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=False),
        sa.Column('tip_time_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('home_team_id', sa.String(10), nullable=False),
        sa.Column('away_team_id', sa.String(10), nullable=False),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(20), server_default='scheduled'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_games_date', 'games', ['game_date'])
    op.create_index('idx_games_status', 'games', ['status'])

    # Odds snapshots table (time-series)
    op.create_table(
        'odds_snapshots',
        sa.Column('snapshot_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(50), nullable=False),
        sa.Column('market_type', sa.String(20), nullable=False),
        sa.Column('book_key', sa.String(50), nullable=False),
        sa.Column('snapshot_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('minutes_to_tip', sa.Integer(), nullable=True),
        # Spread
        sa.Column('home_spread', sa.Numeric(5, 2), nullable=True),
        sa.Column('home_spread_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('away_spread', sa.Numeric(5, 2), nullable=True),
        sa.Column('away_spread_odds', sa.Numeric(6, 3), nullable=True),
        # Moneyline
        sa.Column('home_ml_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('away_ml_odds', sa.Numeric(6, 3), nullable=True),
        # Totals
        sa.Column('total_line', sa.Numeric(5, 2), nullable=True),
        sa.Column('over_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('under_odds', sa.Numeric(6, 3), nullable=True),
        # Implied probs
        sa.Column('home_spread_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('home_ml_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('over_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('is_closing_line', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_odds_game_time', 'odds_snapshots', ['game_id', 'snapshot_time'])
    op.create_index('idx_odds_game_book', 'odds_snapshots', ['game_id', 'book_key'])
    op.create_index('idx_odds_closing', 'odds_snapshots', ['game_id', 'is_closing_line'])
    op.create_index('idx_odds_snapshot_time', 'odds_snapshots', ['snapshot_time'])

    # Markets table
    op.create_table(
        'markets',
        sa.Column('market_id', sa.String(50), primary_key=True),
        sa.Column('game_id', sa.String(50), sa.ForeignKey('games.game_id'), nullable=False),
        sa.Column('market_type', sa.String(20), nullable=False),
        sa.Column('outcome_label', sa.String(100), nullable=False),
        sa.Column('line', sa.Numeric(10, 2), nullable=True),
        sa.Column('odds_decimal', sa.Numeric(5, 3), nullable=False),
        sa.Column('book', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_markets_game', 'markets', ['game_id'])
    op.create_index('idx_markets_type', 'markets', ['market_type'])

    # Model predictions table
    op.create_table(
        'model_predictions',
        sa.Column('prediction_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('market_id', sa.String(50), sa.ForeignKey('markets.market_id'), nullable=False),
        sa.Column('prediction_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('p_ensemble_mean', sa.Numeric(5, 4), nullable=False),
        sa.Column('p_ensemble_std', sa.Numeric(5, 4), nullable=True),
        sa.Column('p_true', sa.Numeric(5, 4), nullable=False),
        sa.Column('p_market', sa.Numeric(5, 4), nullable=False),
        sa.Column('raw_edge', sa.Numeric(6, 4), nullable=False),
        sa.Column('edge_band', sa.String(20), nullable=True),
        sa.Column('features_snapshot_id', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_predictions_market', 'model_predictions', ['market_id'])
    op.create_index('idx_predictions_time', 'model_predictions', ['prediction_time'])

    # Value scores table
    op.create_table(
        'value_scores',
        sa.Column('value_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('prediction_id', sa.BigInteger(), sa.ForeignKey('model_predictions.prediction_id'), nullable=False),
        sa.Column('market_id', sa.String(50), sa.ForeignKey('markets.market_id'), nullable=False),
        sa.Column('calc_time', sa.DateTime(timezone=True), nullable=False),
        # Algorithm A
        sa.Column('algo_a_edge_score', sa.Numeric(6, 4), nullable=True),
        sa.Column('algo_a_confidence', sa.Numeric(5, 3), nullable=True),
        sa.Column('algo_a_market_quality', sa.Numeric(5, 3), nullable=True),
        sa.Column('algo_a_value_score', sa.Numeric(6, 2), nullable=True),
        # Algorithm B
        sa.Column('algo_b_combined_edge', sa.Numeric(6, 4), nullable=True),
        sa.Column('algo_b_confidence', sa.Numeric(5, 3), nullable=True),
        sa.Column('algo_b_market_quality', sa.Numeric(5, 3), nullable=True),
        sa.Column('algo_b_value_score', sa.Numeric(6, 2), nullable=True),
        sa.Column('active_algorithm', sa.String(1), server_default='A'),
        sa.Column('time_to_tip_minutes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_scores_market_calc', 'value_scores', ['market_id', 'calc_time'])
    op.create_index('idx_scores_algo_a', 'value_scores', ['algo_a_value_score'])
    op.create_index('idx_scores_algo_b', 'value_scores', ['algo_b_value_score'])

    # Evaluations table
    op.create_table(
        'evaluations',
        sa.Column('eval_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('value_id', sa.BigInteger(), sa.ForeignKey('value_scores.value_id'), nullable=False),
        sa.Column('game_id', sa.String(50), nullable=False),
        sa.Column('market_id', sa.String(50), nullable=False),
        sa.Column('eval_time', sa.DateTime(timezone=True), nullable=False),
        # Outcome
        sa.Column('home_score', sa.Integer(), nullable=False),
        sa.Column('away_score', sa.Integer(), nullable=False),
        sa.Column('home_margin', sa.Integer(), nullable=False),
        sa.Column('total_points', sa.Integer(), nullable=False),
        # Prediction context
        sa.Column('predicted_prob', sa.Numeric(5, 4), nullable=False),
        sa.Column('market_prob_at_bet', sa.Numeric(5, 4), nullable=False),
        sa.Column('algo_a_score_at_bet', sa.Numeric(6, 2), nullable=True),
        sa.Column('algo_b_score_at_bet', sa.Numeric(6, 2), nullable=True),
        # Result
        sa.Column('bet_won', sa.Boolean(), nullable=False),
        sa.Column('bet_pushed', sa.Boolean(), server_default='false'),
        # CLV
        sa.Column('closing_prob', sa.Numeric(5, 4), nullable=True),
        sa.Column('clv_edge', sa.Numeric(6, 4), nullable=True),
        sa.Column('beat_closing_line', sa.Boolean(), nullable=True),
        # ROI
        sa.Column('odds_at_bet', sa.Numeric(6, 3), nullable=False),
        sa.Column('profit_loss', sa.Numeric(8, 2), nullable=False),
        sa.Column('roi_pct', sa.Numeric(8, 4), nullable=False),
        # Buckets
        sa.Column('value_score_bucket', sa.String(20), nullable=True),
        sa.Column('edge_bucket', sa.String(20), nullable=True),
        sa.Column('confidence_bucket', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_eval_game', 'evaluations', ['game_id'])
    op.create_index('idx_eval_market', 'evaluations', ['market_id'])
    op.create_index('idx_eval_value_id', 'evaluations', ['value_id'])
    op.create_index('idx_eval_bucket', 'evaluations', ['value_score_bucket'])
    op.create_index('idx_eval_time', 'evaluations', ['eval_time'])

    # Users table
    op.create_table(
        'users',
        sa.Column('user_id', sa.String(50), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('tier', sa.String(20), server_default='free'),
        sa.Column('subscription_expires', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('is_verified', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_users_email', 'users', ['email'])


def downgrade() -> None:
    op.drop_table('evaluations')
    op.drop_table('value_scores')
    op.drop_table('model_predictions')
    op.drop_table('markets')
    op.drop_table('odds_snapshots')
    op.drop_table('games')
    op.drop_table('team_stats')
    op.drop_table('teams')
    op.drop_table('users')
