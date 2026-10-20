"""Roster / box-score source for REAL players, joined to Synergy exports.

Synergy's Overall export has no height, position, division, minutes, or
rebounds/assists, so real players need a roster CSV. The workflow:

  1. python -m scripts.load_real_players --pdfs <folder> --template data/roster.csv
     -> one row per exported player, pre-filled with what Synergy implies
        (games, points, FG%, FTA, FT%, turnovers per game)
  2. Fill in the blank columns (height, position, division, minutes, rebounds, ...)
  3. python -m scripts.load_real_players --pdfs <folder> --roster data/roster.csv

Keep roster files under data/ (gitignored): they describe real people.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scout.config import PLAYER_COLUMNS
from scout.data_sources import DataSource, DataSourceError, validate_players
from scout.synergy_pdf import SynergyExport, player_key

ATTENTION_COLUMNS = ["recruiting_stars", "social_followers", "news_mentions_90d"]
OPTIONAL_DEFAULTS = {"conference_strength": 0.5}  # average conference if unknown
ROSTER_COLUMNS = [c for c in PLAYER_COLUMNS if c != "player_id"]
REQUIRED_NUMERIC = [c for c in ROSTER_COLUMNS if c not in
                    {"name", "school", "conference", "division", "position", "class_year"}
                    and c not in ATTENTION_COLUMNS and c not in OPTIONAL_DEFAULTS]

_HEIGHT = re.compile(r"^\s*([5-7])\s*(?:'|’|-|ft|\s)\s*(\d{1,2})?\s*(?:\"|”|in)?\s*$")


def parse_height(value: object) -> float:
    """6'2", 6-2, 6 2, or 74 -> 74 inches. Blank -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return float("nan")
    text = str(value).strip()
    if re.fullmatch(r"\d{2}(\.\d+)?", text):
        return float(text)
    m = _HEIGHT.match(text)
    if not m:
        raise DataSourceError(f"can't read height {text!r} (use 6'2\", 6-2, or 74)")
    return float(int(m.group(1)) * 12 + int(m.group(2) or 0))


class RosterCsvSource(DataSource):
    """Real players from a filled-in roster CSV.

    player_id is derived from name + school, the same key Synergy imports use,
    so roster rows and Synergy tables join automatically. Attention columns are
    optional: blank means "unknown" (not zero), and Underrated is then left blank.
    """

    name = "roster_csv"

    def __init__(self, csv_path: Path | str) -> None:
        self.csv_path = Path(csv_path)

    def fetch_players(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, dtype={"class_year": str})
        missing_cols = [c for c in ROSTER_COLUMNS if c not in df.columns and c not in ATTENTION_COLUMNS
                        and c not in OPTIONAL_DEFAULTS]
        if missing_cols:
            raise DataSourceError(f"roster is missing columns: {missing_cols}")
        df["height_in"] = df["height_in"].map(parse_height)
        for col in ATTENTION_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
        for col, default in OPTIONAL_DEFAULTS.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df else default
        for col in REQUIRED_NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        blanks = [f"{r['name']}: {c}" for _, r in df.iterrows() for c in REQUIRED_NUMERIC if pd.isna(r[c])]
        blanks += [f"{r['name']}: {c}" for _, r in df.iterrows()
                   for c in ("division", "position", "conference") if pd.isna(r[c]) or str(r[c]).strip() == ""]
        if blanks:
            raise DataSourceError("fill in these roster cells first: " + "; ".join(blanks[:20])
                                  + (f" (+{len(blanks) - 20} more)" if len(blanks) > 20 else ""))
        df["division"] = df["division"].str.upper().str.strip()
        df["position"] = df["position"].str.upper().str.strip()
        df["player_id"] = [player_key(n, s) for n, s in zip(df["name"], df["school"])]
        return validate_players(df)


def roster_template(exports: list[SynergyExport]) -> pd.DataFrame:
    """One row per player with an Overall offense export, pre-filled from Synergy.

    Derived from the All Possessions row (approximate, because Synergy rounds):
      points_pg   = PTS / games
      fg_pct      = FG%
      ft_att_pg   = (TSA - FGA) / 0.44 / games
      ft_pct      = (PTS - PPS * FGA) / FTA
      turnovers_pg = TO% * POSS / games
    Everything else is left blank to fill in.
    """
    rows = []
    for exp in exports:
        if exp.view != "overall" or exp.side != "offense":
            continue
        r = exp.table.set_index("situation").loc["All Possessions"]
        g = exp.games or np.nan
        fta = max((r["tsa"] - r["fg_att"]) / 0.44, 0.0)
        ftm = r["pts"] - (r["pps"] * r["fg_att"] if r["fg_att"] else 0.0)
        row = {c: "" for c in ROSTER_COLUMNS}
        row.update({
            "name": exp.player, "school": exp.team, "games": exp.games,
            "points_pg": round(r["pts"] / g, 1), "fg_pct": round(r["fg_pct"], 3),
            "ft_att_pg": round(fta / g, 1), "ft_pct": round(min(max(ftm / fta, 0.0), 1.0), 3) if fta else "",
            "turnovers_pg": round(r["to_pct"] * r["poss"] / g, 1), "conference_strength": 0.5,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=ROSTER_COLUMNS)
