"""Check the scoring against what actually happened to players who moved up to D1.

Input: one row per player who went from D2 / D3 / NAIA to D1, with his last
small-school season ("pre_*" columns) and his first D1 season ("post_*").

Two questions:

1. Are the level factors right? (calibrate_levels)
   The model predicts D1 per-40 production as
       pre_per40 * level_mult(from division, from conference) / conf_mult(to conference)
   For each division we take the median of actual / predicted. A median of 1.0
   means the factor is right; 0.9 means production dropped ~10% more than the
   model assumed, so the factor should be ~10% lower. Bootstrap 95% intervals.
   Only players with enough D1 minutes count, since per-40 rates on 30
   minutes of garbage time are noise. That's survivorship: players who flopped
   so badly they never played drop out, which flatters the factors.

2. Does Translation predict who succeeds? (predictive_check)
   Spearman rank correlation between pre-transfer scores and D1 outcomes:
     - D1 minutes per game: the coaches' verdict, independent of our formulas.
     - D1 production: the same Translation formula applied to his D1 season.
   Compared against naive baselines (raw PPG, and Translation with the level
   adjustment switched off). Translation only earns its keep if it beats them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from scout.config import DIVISIONS, PLAYER_COLUMNS, POSITIONS, SCORING_CONFIG
from scout.data_sources import DataSourceError
from scout.roster import parse_height
from scout.scoring import level_multiplier, score_players

STAT_FIELDS = ["games", "minutes_pg", "points_pg", "rebounds_pg", "assists_pg", "steals_pg", "blocks_pg",
               "turnovers_pg", "fg_pct", "three_pct", "three_att_pg", "ft_pct", "ft_att_pg"]
ID_FIELDS = ["name", "position", "height_in", "from_school", "from_division", "from_conference_strength",
             "to_school", "to_conference_strength", "from_season", "to_season"]
TRANSFER_COLUMNS = ID_FIELDS + [f"pre_{c}" for c in STAT_FIELDS] + [f"post_{c}" for c in STAT_FIELDS]
LOWER_DIVISIONS = [d for d in DIVISIONS if d != "D1"]
MIN_POST_MINUTES = 200  # D1 season minutes needed for per-40 rates to count in calibration
ROTATION_MPG = 15.0     # "became a rotation player"


# --------------------------------------------------------------------------- loading
def load_transfers(path: Path | str) -> pd.DataFrame:
    """Read and validate a transfers CSV. Names every problem it finds."""
    df = pd.read_csv(path)
    missing = [c for c in TRANSFER_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"transfers file is missing columns: {missing}")
    df["height_in"] = df["height_in"].map(parse_height)
    df["position"] = df["position"].str.upper().str.strip()
    df["from_division"] = df["from_division"].str.upper().str.strip()
    problems = []
    bad_div = df.loc[~df["from_division"].isin(LOWER_DIVISIONS), "name"].tolist()
    if bad_div:
        problems.append(f"from_division must be one of {LOWER_DIVISIONS}: {bad_div}")
    bad_pos = df.loc[~df["position"].isin(POSITIONS), "name"].tolist()
    if bad_pos:
        problems.append(f"position must be one of {list(POSITIONS)}: {bad_pos}")
    numeric = [c for c in TRANSFER_COLUMNS if c.startswith(("pre_", "post_"))] + [
        "from_conference_strength", "to_conference_strength"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    blanks = [f"{r['name']}: {c}" for _, r in df.iterrows() for c in numeric
              if pd.isna(r[c]) and not (c.startswith("post_") and r.get("post_games", 1) == 0)]
    if blanks:
        problems.append("blank or non-numeric cells: " + "; ".join(blanks[:15])
                        + (f" (+{len(blanks) - 15} more)" if len(blanks) > 15 else ""))
    if problems:
        raise DataSourceError(" | ".join(problems))
    return df.reset_index(drop=True)


def _season_frame(t: pd.DataFrame, side: str) -> pd.DataFrame:
    """A pre_ or post_ season as a players table the scoring code accepts."""
    out = pd.DataFrame({c: t[f"{side}_{c}"] for c in STAT_FIELDS})
    out["player_id"] = [f"t{i}-{side}" for i in range(len(t))]
    out["name"], out["position"], out["height_in"] = t["name"], t["position"], t["height_in"]
    if side == "pre":
        out["school"], out["division"] = t["from_school"], t["from_division"]
        out["conference_strength"] = t["from_conference_strength"]
    else:
        out["school"], out["division"] = t["to_school"], "D1"
        out["conference_strength"] = t["to_conference_strength"]
    for c in ("conference", "class_year"):
        out[c] = ""
    for c in ("weight_lb", "recruiting_stars", "social_followers", "news_mentions_90d"):
        out[c] = np.nan
    out = out.fillna({c: 0 for c in STAT_FIELDS})  # players who never got on the floor at D1
    return out[list(PLAYER_COLUMNS)]


@contextmanager
def no_level_adjustment() -> Iterator[None]:
    """Temporarily score every division as if it were an average D1 conference."""
    saved = {k: SCORING_CONFIG[k] for k in ("level_factor", "conference_slope")}
    SCORING_CONFIG["level_factor"] = {d: 1.0 for d in DIVISIONS}
    SCORING_CONFIG["conference_slope"] = 0.0
    try:
        yield
    finally:
        SCORING_CONFIG.update(saved)


def score_transfers(t: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add predictors (pre-transfer scores) and outcomes (D1 results) to each transfer row.

    reference: D1 players used as the percentile baseline (box-score columns only).
    """
    ref = reference[reference["division"] == "D1"][list(PLAYER_COLUMNS)]
    pre, post = _season_frame(t, "pre"), _season_frame(t, "post")
    out = t.copy()
    out["translation"] = score_players(pre, reference=ref)["translation_score"].to_numpy()
    with no_level_adjustment():
        out["translation_no_level"] = score_players(pre, reference=ref)["translation_score"].to_numpy()
    out["raw_ppg"] = t["pre_points_pg"]
    played = post["minutes_pg"] > 0
    post_scored = score_players(post[played], reference=ref)
    out["d1_production"] = np.nan
    out.loc[played.to_numpy(), "d1_production"] = post_scored["translation_score"].to_numpy()
    out["d1_minutes_pg"] = t["post_minutes_pg"].fillna(0)
    out["d1_season_minutes"] = (t["post_games"] * t["post_minutes_pg"]).fillna(0)
    return out


# --------------------------------------------------------------------------- statistics
def spearman(x: pd.Series, y: pd.Series) -> float:
    """Rank correlation (pandas needs scipy for this; ranks + Pearson is the same thing)."""
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def bootstrap(stat, n: int, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(idx) over resampled row indices."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(reps)]
    vals = [v for v in vals if not np.isnan(v)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def _production_per40(df: pd.DataFrame, side: str) -> pd.Series:
    """Points + rebounds + assists per 40 minutes."""
    pra = df[f"{side}_points_pg"] + df[f"{side}_rebounds_pg"] + df[f"{side}_assists_pg"]
    return pra * 40 / df[f"{side}_minutes_pg"].replace(0, np.nan)


@dataclass
class LevelCalibration:
    division: str
    n: int
    configured: float
    median_ratio: float  # actual / predicted D1 production; 1.0 = factor is right
    ci: tuple[float, float]
    suggested: float
    suggested_ci: tuple[float, float]


def calibrate_levels(t: pd.DataFrame) -> list[LevelCalibration]:
    """Per division: how much production actually carried over vs what the factors predict."""
    cfg = SCORING_CONFIG
    usable = t[(t["post_games"] * t["post_minutes_pg"]).fillna(0) >= MIN_POST_MINUTES].copy()
    to_conf = cfg["conference_base"] + cfg["conference_slope"] * usable["to_conference_strength"]
    predicted = (_production_per40(usable, "pre")
                 * level_multiplier(usable["from_division"], usable["from_conference_strength"]) / to_conf)
    usable["ratio"] = _production_per40(usable, "post") / predicted
    out = []
    for div in LOWER_DIVISIONS:
        r = usable.loc[usable["from_division"] == div, "ratio"].dropna().to_numpy()
        if len(r) == 0:
            continue
        med = float(np.median(r))
        ci = bootstrap(lambda idx: float(np.median(r[idx])), len(r)) if len(r) >= 5 else (np.nan, np.nan)
        conf = cfg["level_factor"][div]
        out.append(LevelCalibration(div, len(r), conf, med, ci, round(conf * med, 3),
                                    (round(conf * ci[0], 3), round(conf * ci[1], 3))))
    return out


@dataclass
class PredictiveResult:
    predictor: str
    outcome: str
    n: int
    rho: float
    ci: tuple[float, float]


PREDICTORS = {"translation": "Translation (full model)",
              "translation_no_level": "Translation, no level adjustment",
              "raw_ppg": "Raw points per game"}
OUTCOMES = {"d1_minutes_pg": "D1 minutes per game", "d1_production": "D1 production (same formula, D1 season)"}


def predictive_check(scored: pd.DataFrame) -> list[PredictiveResult]:
    """Spearman correlation of each predictor with each D1 outcome, with bootstrap intervals."""
    out = []
    for o in OUTCOMES:
        for p in PREDICTORS:
            d = scored[[p, o]].dropna().reset_index(drop=True)
            rho = spearman(d[p], d[o])
            ci = bootstrap(lambda idx: spearman(d[p].iloc[idx].reset_index(drop=True),
                                                d[o].iloc[idx].reset_index(drop=True)), len(d))
            out.append(PredictiveResult(p, o, len(d), rho, ci))
    return out


def rotation_hit_rates(scored: pd.DataFrame) -> pd.DataFrame:
    """Share who became D1 rotation players (15+ mpg), by pre-transfer Translation third."""
    s = scored.copy()
    s["third"] = pd.qcut(s["translation"].rank(method="first"), 3, labels=["Bottom third", "Middle third", "Top third"])
    g = s.groupby("third", observed=True)
    return pd.DataFrame({"players": g.size(), "rotation_rate": g["d1_minutes_pg"].apply(lambda m: (m >= ROTATION_MPG).mean()),
                         "avg_translation": g["translation"].mean()}).iloc[::-1]


# --------------------------------------------------------------------------- self-test data
TRUE_FACTORS = {"D2": 0.72, "D3": 0.58, "NAIA": 0.66}  # deliberately NOT the configured values


def simulate_transfers(players: pd.DataFrame, seed: int = 7, true_factors: dict[str, float] = TRUE_FACTORS) -> pd.DataFrame:
    """Fake move-ups from a world with KNOWN level factors, to check the harness can recover them.

    Takes non-D1 players who out-produced their division (the kind who move up), and
    gives each a D1 season: per-40 production scaled by the true factor and destination
    conference, efficiency down a bit, and minutes earned by how well he actually produces.
    """
    rng = np.random.default_rng(seed)
    p = players[players["division"] != "D1"].copy()
    p["pra40"] = (p["points_pg"] + p["rebounds_pg"] + p["assists_pg"]) * 40 / p["minutes_pg"]
    p = p[p["pra40"] > p.groupby("division")["pra40"].transform("median")].reset_index(drop=True)
    cfg = SCORING_CONFIG
    to_conf = rng.uniform(0.2, 0.9, len(p))
    true_mult = (p["division"].map(true_factors).to_numpy()
                 * (cfg["conference_base"] + cfg["conference_slope"] * p["conference_strength"].to_numpy())
                 / (cfg["conference_base"] + cfg["conference_slope"] * to_conf))
    carry = true_mult * rng.lognormal(0, 0.15, len(p))  # how much per-40 production survives the move
    quality = pd.Series(p["pra40"].to_numpy() * carry).rank(pct=True).to_numpy()
    mpg = np.clip(6 + 24 * quality + rng.normal(0, 5, len(p)), 0, 36)
    mpg[rng.random(len(p)) < 0.08] = 0  # some never crack the rotation at all
    games = np.where(mpg > 0, rng.integers(20, 33, len(p)), 0)
    eff_drop = (1 - p["division"].map(true_factors).to_numpy()) * 0.12
    t = pd.DataFrame({
        "name": p["name"], "position": p["position"], "height_in": p["height_in"],
        "from_school": p["school"], "from_division": p["division"], "from_conference_strength": p["conference_strength"],
        "to_school": [f"D1 School {i}" for i in range(len(p))], "to_conference_strength": to_conf.round(2),
        "from_season": "sim-1", "to_season": "sim-2",
    })
    for c in STAT_FIELDS:
        t[f"pre_{c}"] = p[c].to_numpy()
    scale = mpg / p["minutes_pg"].to_numpy()  # per-game volume moves with minutes
    t["post_games"], t["post_minutes_pg"] = games, mpg.round(1)
    for c in ("points_pg", "rebounds_pg", "assists_pg", "steals_pg", "blocks_pg"):
        t[f"post_{c}"] = (p[c].to_numpy() * carry * scale).round(1)
    t["post_turnovers_pg"] = (p["turnovers_pg"].to_numpy() / carry * scale * 0.9).round(1)
    t["post_three_att_pg"] = (p["three_att_pg"].to_numpy() * scale).round(1)
    t["post_ft_att_pg"] = (p["ft_att_pg"].to_numpy() * carry * scale).round(1)
    t["post_fg_pct"] = (p["fg_pct"].to_numpy() - eff_drop + rng.normal(0, 0.02, len(p))).clip(0.25, 0.75).round(3)
    t["post_three_pct"] = (p["three_pct"].to_numpy() - eff_drop * 0.7 + rng.normal(0, 0.03, len(p))).clip(0.15, 0.5).round(3)
    t["post_ft_pct"] = (p["ft_pct"].to_numpy() + rng.normal(0, 0.03, len(p))).clip(0.4, 0.95).round(3)
    zero = t["post_games"] == 0
    t.loc[zero, [f"post_{c}" for c in STAT_FIELDS]] = 0
    return t[TRANSFER_COLUMNS]
