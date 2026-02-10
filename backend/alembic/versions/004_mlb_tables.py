"""Add MLB tables for baseball betting support.

Revision ID: 004
Revises: 003
Create Date: 2026-02-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MLB Teams table
    op.create_table(
        'mlb_teams',
        sa.Column('team_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('team_abbr', sa.String(5), unique=True, nullable=False),
        sa.Column('team_name', sa.String(100), nullable=False),
        sa.Column('league', sa.String(2), nullable=False),  # AL, NL
        sa.Column('division', sa.String(10), nullable=False),  # East, Central, West
        sa.Column('external_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # MLB Pitchers table
    op.create_table(
        'mlb_pitchers',
        sa.Column('pitcher_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('player_name', sa.String(100), nullable=False),
        sa.Column('team_abbr', sa.String(5), nullable=True),
        sa.Column('throws', sa.String(1), nullable=True),  # L, R
        sa.Column('external_id', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_mlb_pitchers_external_id', 'mlb_pitchers', ['external_id'])

    # MLB Pitcher Stats table (rolling stats)
    op.create_table(
        'mlb_pitcher_stats',
        sa.Column('stat_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('pitcher_id', sa.Integer(), sa.ForeignKey('mlb_pitchers.pitcher_id'), nullable=False),
        sa.Column('stat_date', sa.Date(), nullable=False),
        # Core pitching stats
        sa.Column('era', sa.Numeric(5, 2), nullable=True),
        sa.Column('whip', sa.Numeric(5, 3), nullable=True),
        sa.Column('k_per_9', sa.Numeric(5, 2), nullable=True),
        sa.Column('bb_per_9', sa.Numeric(5, 2), nullable=True),
        sa.Column('fip', sa.Numeric(5, 2), nullable=True),
        # Workload
        sa.Column('innings_pitched', sa.Numeric(6, 1), nullable=True),
        sa.Column('games_started', sa.Integer(), nullable=True),
        # Advanced metrics
        sa.Column('k_pct', sa.Numeric(5, 3), nullable=True),
        sa.Column('bb_pct', sa.Numeric(5, 3), nullable=True),
        sa.Column('hr_per_9', sa.Numeric(5, 2), nullable=True),
        # Recent form
        sa.Column('era_l5', sa.Numeric(5, 2), nullable=True),
        sa.Column('whip_l5', sa.Numeric(5, 3), nullable=True),
        # Composite score
        sa.Column('quality_score', sa.Numeric(5, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_mlb_pitcher_stats_pitcher', 'mlb_pitcher_stats', ['pitcher_id'])
    op.create_index('idx_mlb_pitcher_stats_date', 'mlb_pitcher_stats', ['stat_date'])
    op.create_index('idx_mlb_pitcher_stats_unique', 'mlb_pitcher_stats', ['pitcher_id', 'stat_date'], unique=True)

    # MLB Team Stats table (rolling stats)
    op.create_table(
        'mlb_team_stats',
        sa.Column('stat_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('team_abbr', sa.String(5), nullable=False),
        sa.Column('stat_date', sa.Date(), nullable=False),
        # Offensive stats
        sa.Column('runs_per_game', sa.Numeric(5, 2), nullable=True),
        sa.Column('runs_allowed_per_game', sa.Numeric(5, 2), nullable=True),
        sa.Column('run_diff_per_game', sa.Numeric(6, 2), nullable=True),
        # Batting metrics
        sa.Column('ops', sa.Numeric(5, 3), nullable=True),
        sa.Column('batting_avg', sa.Numeric(5, 3), nullable=True),
        sa.Column('obp', sa.Numeric(5, 3), nullable=True),
        sa.Column('slg', sa.Numeric(5, 3), nullable=True),
        # Pitching staff
        sa.Column('era', sa.Numeric(5, 2), nullable=True),
        sa.Column('bullpen_era', sa.Numeric(5, 2), nullable=True),
        # Win/Loss
        sa.Column('wins', sa.Integer(), nullable=True),
        sa.Column('losses', sa.Integer(), nullable=True),
        sa.Column('win_pct', sa.Numeric(4, 3), nullable=True),
        # Home/Away splits
        sa.Column('home_wins', sa.Integer(), nullable=True),
        sa.Column('home_losses', sa.Integer(), nullable=True),
        sa.Column('home_win_pct', sa.Numeric(4, 3), nullable=True),
        sa.Column('away_wins', sa.Integer(), nullable=True),
        sa.Column('away_losses', sa.Integer(), nullable=True),
        sa.Column('away_win_pct', sa.Numeric(4, 3), nullable=True),
        # Recent form
        sa.Column('last_10_wins', sa.Integer(), nullable=True),
        sa.Column('last_10_losses', sa.Integer(), nullable=True),
        sa.Column('last_10_record', sa.String(10), nullable=True),
        # Runs - rolling windows
        sa.Column('runs_per_game_l10', sa.Numeric(5, 2), nullable=True),
        sa.Column('runs_allowed_per_game_l10', sa.Numeric(5, 2), nullable=True),
        # Rest and schedule
        sa.Column('days_rest', sa.Integer(), nullable=True),
        sa.Column('games_last_7_days', sa.Integer(), nullable=True),
        # Betting records
        sa.Column('ats_wins', sa.Integer(), nullable=True),
        sa.Column('ats_losses', sa.Integer(), nullable=True),
        sa.Column('ats_pushes', sa.Integer(), nullable=True),
        sa.Column('ou_overs', sa.Integer(), nullable=True),
        sa.Column('ou_unders', sa.Integer(), nullable=True),
        sa.Column('ou_pushes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_mlb_team_stats_team', 'mlb_team_stats', ['team_abbr'])
    op.create_index('idx_mlb_team_stats_date', 'mlb_team_stats', ['stat_date'])
    op.create_index('idx_mlb_team_stats_unique', 'mlb_team_stats', ['team_abbr', 'stat_date'], unique=True)

    # MLB Games table
    op.create_table(
        'mlb_games',
        sa.Column('game_id', sa.String(50), primary_key=True),
        sa.Column('game_date', sa.Date(), nullable=False),
        sa.Column('game_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('home_team', sa.String(5), nullable=False),
        sa.Column('away_team', sa.String(5), nullable=False),
        sa.Column('home_starter_id', sa.Integer(), sa.ForeignKey('mlb_pitchers.pitcher_id'), nullable=True),
        sa.Column('away_starter_id', sa.Integer(), sa.ForeignKey('mlb_pitchers.pitcher_id'), nullable=True),
        sa.Column('status', sa.String(20), server_default='scheduled'),
        sa.Column('inning', sa.Integer(), nullable=True),
        sa.Column('inning_state', sa.String(10), nullable=True),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('external_id', sa.String(50), nullable=True),
        sa.Column('season', sa.Integer(), nullable=True),
        sa.Column('game_type', sa.String(5), nullable=True),  # R, P, S
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_mlb_games_date', 'mlb_games', ['game_date'])
    op.create_index('idx_mlb_games_external_id', 'mlb_games', ['external_id'])

    # MLB Game Context table (weather, park factors)
    op.create_table(
        'mlb_game_context',
        sa.Column('context_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(50), nullable=False, unique=True),
        sa.Column('venue_name', sa.String(100), nullable=True),
        sa.Column('venue_id', sa.Integer(), nullable=True),
        sa.Column('park_factor', sa.Numeric(4, 2), nullable=True),
        sa.Column('temperature', sa.Integer(), nullable=True),
        sa.Column('wind_speed', sa.Integer(), nullable=True),
        sa.Column('wind_direction', sa.String(20), nullable=True),
        sa.Column('humidity', sa.Integer(), nullable=True),
        sa.Column('precipitation_pct', sa.Integer(), nullable=True),
        sa.Column('sky_condition', sa.String(20), nullable=True),
        sa.Column('is_dome', sa.Boolean(), server_default='false'),
        sa.Column('is_retractable', sa.Boolean(), server_default='false'),
        sa.Column('roof_status', sa.String(10), nullable=True),
        sa.Column('wind_factor', sa.Numeric(4, 2), nullable=True),
        sa.Column('weather_factor', sa.Numeric(4, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_mlb_game_context_game', 'mlb_game_context', ['game_id'])

    # MLB Markets table (betting lines)
    op.create_table(
        'mlb_markets',
        sa.Column('market_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(50), sa.ForeignKey('mlb_games.game_id'), nullable=False),
        sa.Column('market_type', sa.String(20), nullable=False),  # moneyline, runline, total
        sa.Column('line', sa.Numeric(5, 1), nullable=True),
        sa.Column('home_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('away_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('over_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('under_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('book', sa.String(50), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_mlb_markets_game', 'mlb_markets', ['game_id'])

    # MLB Predictions table
    op.create_table(
        'mlb_predictions',
        sa.Column('prediction_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(50), sa.ForeignKey('mlb_games.game_id'), nullable=False),
        sa.Column('market_type', sa.String(20), nullable=False),
        sa.Column('predicted_run_diff', sa.Numeric(5, 2), nullable=True),
        sa.Column('predicted_total', sa.Numeric(5, 1), nullable=True),
        sa.Column('p_home_win', sa.Numeric(4, 3), nullable=True),
        sa.Column('p_away_win', sa.Numeric(4, 3), nullable=True),
        sa.Column('p_home_cover', sa.Numeric(4, 3), nullable=True),
        sa.Column('p_away_cover', sa.Numeric(4, 3), nullable=True),
        sa.Column('p_over', sa.Numeric(4, 3), nullable=True),
        sa.Column('p_under', sa.Numeric(4, 3), nullable=True),
        sa.Column('market_implied_prob', sa.Numeric(4, 3), nullable=True),
        sa.Column('edge', sa.Numeric(6, 2), nullable=True),
        sa.Column('edge_pct', sa.Numeric(6, 2), nullable=True),
        sa.Column('value_score', sa.Numeric(5, 1), nullable=True),
        sa.Column('confidence', sa.String(20), nullable=True),
        sa.Column('recommendation', sa.String(30), nullable=True),
        sa.Column('model_version', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_mlb_predictions_game', 'mlb_predictions', ['game_id'])

    # MLB Prediction Snapshots table (for tracking)
    op.create_table(
        'mlb_prediction_snapshots',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(100), nullable=False),
        sa.Column('snapshot_time', sa.DateTime(timezone=True), nullable=False),
        # Game info
        sa.Column('home_team', sa.String(5), nullable=False),
        sa.Column('away_team', sa.String(5), nullable=False),
        sa.Column('game_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=True),
        # Starters
        sa.Column('home_starter_name', sa.String(100), nullable=True),
        sa.Column('away_starter_name', sa.String(100), nullable=True),
        sa.Column('home_starter_era', sa.Numeric(5, 2), nullable=True),
        sa.Column('away_starter_era', sa.Numeric(5, 2), nullable=True),
        # Winner prediction
        sa.Column('predicted_winner', sa.String(5), nullable=False),
        sa.Column('winner_probability', sa.Numeric(5, 3), nullable=False),
        sa.Column('winner_confidence', sa.String(20), nullable=False),
        # Run predictions
        sa.Column('predicted_run_diff', sa.Numeric(5, 2), nullable=True),
        sa.Column('predicted_total', sa.Numeric(5, 1), nullable=True),
        # Best ML bet
        sa.Column('best_ml_team', sa.String(5), nullable=True),
        sa.Column('best_ml_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('best_ml_value_score', sa.Integer(), nullable=True),
        sa.Column('best_ml_edge', sa.Numeric(5, 2), nullable=True),
        # Best RL bet
        sa.Column('best_rl_team', sa.String(5), nullable=True),
        sa.Column('best_rl_line', sa.Numeric(5, 1), nullable=True),
        sa.Column('best_rl_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('best_rl_value_score', sa.Integer(), nullable=True),
        sa.Column('best_rl_edge', sa.Numeric(5, 2), nullable=True),
        # Best total bet
        sa.Column('best_total_direction', sa.String(10), nullable=True),
        sa.Column('best_total_line', sa.Numeric(5, 1), nullable=True),
        sa.Column('best_total_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('best_total_value_score', sa.Integer(), nullable=True),
        sa.Column('best_total_edge', sa.Numeric(5, 2), nullable=True),
        # Overall best bet
        sa.Column('best_bet_type', sa.String(20), nullable=True),
        sa.Column('best_bet_team', sa.String(10), nullable=True),
        sa.Column('best_bet_line', sa.Numeric(5, 1), nullable=True),
        sa.Column('best_bet_odds', sa.Numeric(6, 3), nullable=True),
        sa.Column('best_bet_value_score', sa.Integer(), nullable=True),
        sa.Column('best_bet_edge', sa.Numeric(5, 2), nullable=True),
        # Context
        sa.Column('venue_name', sa.String(100), nullable=True),
        sa.Column('park_factor', sa.Numeric(4, 2), nullable=True),
        sa.Column('temperature', sa.Integer(), nullable=True),
        sa.Column('is_dome', sa.Boolean(), nullable=True),
        # Line movement
        sa.Column('opening_ml_home', sa.Numeric(6, 3), nullable=True),
        sa.Column('current_ml_home', sa.Numeric(6, 3), nullable=True),
        sa.Column('opening_total', sa.Numeric(5, 1), nullable=True),
        sa.Column('current_total', sa.Numeric(5, 1), nullable=True),
        # Actual results (filled after game)
        sa.Column('actual_winner', sa.String(5), nullable=True),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        # Grading
        sa.Column('winner_correct', sa.Boolean(), nullable=True),
        sa.Column('best_ml_result', sa.String(20), nullable=True),
        sa.Column('best_ml_profit', sa.Numeric(8, 2), nullable=True),
        sa.Column('best_rl_result', sa.String(20), nullable=True),
        sa.Column('best_rl_profit', sa.Numeric(8, 2), nullable=True),
        sa.Column('best_total_result', sa.String(20), nullable=True),
        sa.Column('best_total_profit', sa.Numeric(8, 2), nullable=True),
        sa.Column('best_bet_result', sa.String(20), nullable=True),
        sa.Column('best_bet_profit', sa.Numeric(8, 2), nullable=True),
        # Factors
        sa.Column('factors', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('idx_mlb_prediction_snapshots_game', 'mlb_prediction_snapshots', ['game_id'])
    op.create_index('idx_mlb_prediction_snapshots_date', 'mlb_prediction_snapshots', ['game_date'])


def downgrade() -> None:
    op.drop_table('mlb_prediction_snapshots')
    op.drop_table('mlb_predictions')
    op.drop_table('mlb_markets')
    op.drop_table('mlb_game_context')
    op.drop_table('mlb_games')
    op.drop_table('mlb_team_stats')
    op.drop_table('mlb_pitcher_stats')
    op.drop_table('mlb_pitchers')
    op.drop_table('mlb_teams')
