"""Central configuration: paths, env vars, and domain constants.

Everything tunable lives here so there is one place to look when a number
seems off. Scoring constants are added in Phase 2.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.getenv("SCOUT_DB_PATH", DATA_DIR / "scout.duckdb"))
if not DB_PATH.is_absolute():
    DB_PATH = PROJECT_ROOT / DB_PATH
SEED_PLAYERS_CSV = DATA_DIR / "seed_players.csv"
SEED_PLANTED_CSV = DATA_DIR / "seed_planted.csv"

SCOUT_MODEL = os.getenv("SCOUT_MODEL", "claude-sonnet-5-5")
API_URL = os.getenv("SCOUT_API_URL", "http://localhost:8000")

DIVISIONS = ("D1", "D2", "D3", "NAIA")
POSITIONS = ("PG", "SG", "SF", "PF", "C")
CLASS_YEARS = ("FR", "SO", "JR", "SR")

# Column contract every DataSource must satisfy. `conference` is an extra
# beyond the original spec: it makes conference_strength explainable in the UI.
PLAYER_COLUMNS = (
    "player_id", "name", "school", "conference", "division", "conference_strength",
    "position", "height_in", "weight_lb", "class_year", "games", "minutes_pg",
    "points_pg", "rebounds_pg", "assists_pg", "steals_pg", "blocks_pg", "turnovers_pg",
    "fg_pct", "three_pct", "three_att_pg", "ft_pct", "ft_att_pg",
    "recruiting_stars", "social_followers", "news_mentions_90d",
)


def has_api_key() -> bool:
    """True if an Anthropic key is configured (never returns or logs the key)."""
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Scoring assumptions. ALL of these are judgment calls to tune, not facts.
# ---------------------------------------------------------------------------
SCORING_CONFIG: dict = {
    # How much a season of production at each level is worth at D1.
    "level_factor": {"D1": 1.00, "D2": 0.80, "D3": 0.65, "NAIA": 0.70},
    # Conference adjustment: multiplier = base + slope * conference_strength.
    # An average conference (0.5) gives exactly 1.0; range is 0.85-1.15.
    "conference_base": 0.85,
    "conference_slope": 0.30,
    # 3P% is shrunk toward the D1 position average by this many "phantom"
    # attempts, so a center who went 3-for-5 doesn't look like an elite shooter.
    "three_pct_prior_attempts": 50,
    # Translation scores for low-minute players are pulled toward this prior.
    # Full trust at >= full_trust_minutes season minutes.
    "reliability_prior": 40.0,
    "full_trust_minutes": 600,
    # Attention blend (weights sum to 1).
    "attention_weights": {"recruiting_stars": 0.40, "social_followers": 0.35, "news_mentions_90d": 0.25},
}

# Position-aware Translation weights over adjusted stats (each row sums to 1).
# PG emphasizes decision-making and shooting; bigs emphasize rebounding/rim protection.
POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "PG": {"ast_to": 0.22, "ast_per40": 0.20, "ts_pct": 0.18, "three_pct_adj": 0.15,
           "pts_per40": 0.10, "stl_per40": 0.08, "ft_pct": 0.07},
    "SG": {"ts_pct": 0.22, "three_pct_adj": 0.20, "pts_per40": 0.20, "ast_per40": 0.10,
           "ast_to": 0.10, "stl_per40": 0.08, "ft_pct": 0.10},
    "SF": {"ts_pct": 0.20, "pts_per40": 0.18, "three_pct_adj": 0.14, "reb_per40": 0.14,
           "stl_per40": 0.10, "ast_to": 0.10, "blk_per40": 0.06, "ft_pct": 0.08},
    "PF": {"ts_pct": 0.20, "reb_per40": 0.22, "pts_per40": 0.16, "blk_per40": 0.14,
           "three_pct_adj": 0.10, "stl_per40": 0.08, "ast_to": 0.10},
    "C": {"ts_pct": 0.22, "reb_per40": 0.25, "blk_per40": 0.22, "pts_per40": 0.15,
          "ft_pct": 0.06, "ast_to": 0.10},
}
