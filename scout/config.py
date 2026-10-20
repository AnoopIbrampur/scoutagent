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
