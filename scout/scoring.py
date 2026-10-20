"""Transparent scoring: every number shown in the UI can be traced to a formula here.

Pipeline (each step adds columns to the DataFrame):
  1. derive_stats      - TS%, AST/TO, per-40 rates, volume-shrunk 3P%
  2. adjust_stats      - small-sample shrinkage + level/conference adjustment -> adj_* columns
  3. add_percentiles   - pct_* = percentile of adj_* among D1 players at the same position.
                         Always "higher = better": for lower-is-better stats (turnovers,
                         points allowed) pct_* is how many D1 players he BEATS.
  4. translation       - position-weighted blend of pct_*, shrunk for low minutes
  5. attention         - percentile blend of stars, log followers, news mentions
  6. underrated        - (translation - attention), rescaled to 0-100
  explain_player()     - top 3 factors + top concern, as plain-English strings

Synergy stats (off_ppp, ssm, to_pct, def_ppp) come from scout.synergy.attach_synergy.
Players without them are scored on box-score stats only.
All tunable constants live in scout.config (SCORING_CONFIG, POSITION_WEIGHTS).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scout.config import POSITION_WEIGHTS, POSITIONS, SCORING_CONFIG

COUNTING_STATS = ["pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40"]
EFFICIENCY_STATS = ["ts_pct", "three_pct_adj", "ft_pct", "ast_to"]
NEGATIVE_STATS = ["tov_per40"]  # lower is better; used for concerns, not weights
SYNERGY_STATS = ["off_ppp", "ssm", "to_pct", "def_ppp"]
ALL_SCORED_STATS = COUNTING_STATS + EFFICIENCY_STATS + NEGATIVE_STATS + SYNERGY_STATS
LOWER_IS_BETTER = {"tov_per40", "to_pct", "def_ppp"}

STAT_LABELS = {
    "pts_per40": "Points per 40", "reb_per40": "Rebounds per 40", "ast_per40": "Assists per 40",
    "stl_per40": "Steals per 40", "blk_per40": "Blocks per 40", "tov_per40": "Turnovers per 40",
    "ts_pct": "True shooting %", "three_pct_adj": "3P% (volume-adjusted)", "ft_pct": "FT%",
    "ast_to": "Assist-to-turnover ratio",
    "off_ppp": "Points per possession (Synergy)", "ssm": "Shot-making over expected (Synergy SSM)",
    "to_pct": "Turnover rate (Synergy TO%)", "def_ppp": "Points allowed per possession (Synergy)",
}
PCT_STATS = {"ts_pct", "three_pct_adj", "ft_pct", "to_pct"}
# How a lower-is-better stat reads when it is a concern ("... than 72% of D1 PGs").
_WORSE_PHRASE = {"tov_per40": "higher than", "to_pct": "turns it over more often than",
                 "def_ppp": "allows more per possession than"}


# --------------------------------------------------------------------------- 1
def estimate_fga(df: pd.DataFrame) -> pd.Series:
    """Approximate field goal attempts per game from points and percentages.

    The data has no FGA column, so we invert the points identity:
        PTS = 2*FGM + 3PM + FTM      (each 3 is a FGM worth one extra point)
    =>  FGM = (PTS - FTM - 3PM) / 2,   FGA = FGM / FG%
    where FTM = FTA * FT% and 3PM = 3PA * 3P%. Rounding in the per-game inputs
    makes this approximate (typically within ~0.2 attempts).
    """
    ftm = df["ft_att_pg"] * df["ft_pct"]
    tpm = df["three_att_pg"] * df["three_pct"]
    fgm = ((df["points_pg"] - ftm - tpm) / 2).clip(lower=0)
    fga = fgm / df["fg_pct"].replace(0, np.nan)
    # FGA can never be below the 3PA it contains.
    return fga.fillna(0).clip(lower=df["three_att_pg"])


def true_shooting(points: pd.Series, fga: pd.Series, fta: pd.Series) -> pd.Series:
    """TS% = PTS / (2 * (FGA + 0.44 * FTA)). Returns 0 when there are no attempts."""
    denom = 2 * (fga + 0.44 * fta)
    return (points / denom.replace(0, np.nan)).fillna(0.0)


def derive_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Add fga_pg, ts_pct, ast_to, *_per40, and three_pct_adj columns.

    Synergy columns are added as NaN if missing, so every later step can rely on them.
    """
    df = players.copy()
    df["fga_pg"] = estimate_fga(df)
    df["ts_pct"] = true_shooting(df["points_pg"], df["fga_pg"], df["ft_att_pg"])
    df["ast_to"] = df["assists_pg"] / df["turnovers_pg"].clip(lower=0.1)
    per40 = 40.0 / df["minutes_pg"].clip(lower=1.0)
    for src, dst in [("points_pg", "pts_per40"), ("rebounds_pg", "reb_per40"),
                     ("assists_pg", "ast_per40"), ("steals_pg", "stl_per40"),
                     ("blocks_pg", "blk_per40"), ("turnovers_pg", "tov_per40")]:
        df[dst] = df[src] * per40
    df["three_pct_adj"] = shrink_three_pct(df)
    df["season_minutes"] = df["games"] * df["minutes_pg"]
    for col in SYNERGY_STATS + ["syn_off_poss", "syn_def_poss", "syn_fga"]:
        if col not in df:
            df[col] = np.nan
    return df


def shrink_three_pct(df: pd.DataFrame) -> pd.Series:
    """Shrink 3P% toward the D1 position average using season attempts (empirical Bayes).

    adj = (makes + k * prior) / (attempts + k), with k = three_pct_prior_attempts.
    A 200-attempt shooter barely moves; a 10-attempt center lands near average.
    """
    k = SCORING_CONFIG["three_pct_prior_attempts"]
    attempts = df["three_att_pg"] * df["games"]
    makes = attempts * df["three_pct"]
    d1 = df[df["division"] == "D1"]
    prior_by_pos = (d1["three_att_pg"] * d1["games"] * d1["three_pct"]).groupby(d1["position"]).sum() / (
        (d1["three_att_pg"] * d1["games"]).groupby(d1["position"]).sum()
    )
    prior = df["position"].map(prior_by_pos).fillna(df["three_pct"].mean())
    return (makes + k * prior) / (attempts + k)


def shrink_to_d1_average(df: pd.DataFrame, stat: str, sample: str, k: float) -> pd.Series:
    """(x * n + avg * k) / (n + k): small Synergy samples are pulled to the D1 position average."""
    d1 = df[(df["division"] == "D1") & df[stat].notna()]
    avg = df["position"].map(d1.groupby("position")[stat].mean())
    n = df[sample].fillna(0)
    return (df[stat] * n + avg * k) / (n + k)


# --------------------------------------------------------------------------- 2
def level_multiplier(division: str | pd.Series, conference_strength: float | pd.Series) -> Any:
    """Production multiplier = level_factor[division] * (base + slope * conference_strength)."""
    cfg = SCORING_CONFIG
    factor = (pd.Series(division).map(cfg["level_factor"]).to_numpy()
              if isinstance(division, pd.Series) else cfg["level_factor"][division])
    conf = cfg["conference_base"] + cfg["conference_slope"] * np.asarray(conference_strength)
    return factor * conf


def _discount_good_side(x: pd.Series, avg: pd.Series, trust: pd.Series, lower_is_better: bool) -> np.ndarray:
    """Pull only the better-than-D1-average part of x toward the average by `trust`."""
    better = (x < avg) if lower_is_better else (x > avg)
    return np.where(better, avg + (x - avg) * trust, x)


def adjust_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add level_mult and adj_* columns.

    * Counting rates (per 40) are multiplied by the level multiplier.
    * Turnovers per 40 are divided by it (weaker defenses force fewer turnovers,
      so a small-school turnover rate understates the D1 rate).
    * Efficiency stats can't be scaled (0.60 TS% * 0.8 = 0.48 is nonsense), so
      only the part BETTER than the D1 position average is discounted:
          adj = avg + (x - avg) * min(mult, 1)   if x is better than avg, else x.
      Strong numbers against weak competition are trusted less; weak numbers
      get no free pass. Synergy stats are treated the same way after
      small-sample shrinkage (PPP allowed is "better" when lower).
    """
    df = df.copy()
    df["level_mult"] = level_multiplier(df["division"], df["conference_strength"])
    for stat in COUNTING_STATS:
        df[f"adj_{stat}"] = df[stat] * df["level_mult"]
    df["adj_tov_per40"] = df["tov_per40"] / df["level_mult"]

    trust = df["level_mult"].clip(upper=1.0)
    d1_avg = df[df["division"] == "D1"].groupby("position")[EFFICIENCY_STATS].mean()
    for stat in EFFICIENCY_STATS:
        df[f"adj_{stat}"] = _discount_good_side(df[stat], df["position"].map(d1_avg[stat]), trust, False)

    cfg = SCORING_CONFIG
    shrunk = {
        "off_ppp": shrink_to_d1_average(df, "off_ppp", "syn_off_poss", cfg["synergy_prior_off_poss"]),
        "ssm": shrink_to_d1_average(df, "ssm", "syn_fga", cfg["synergy_prior_off_poss"]),
        "to_pct": shrink_to_d1_average(df, "to_pct", "syn_off_poss", cfg["synergy_prior_off_poss"]),
        "def_ppp": shrink_to_d1_average(df, "def_ppp", "syn_def_poss", cfg["synergy_prior_def_poss"]),
    }
    for stat, x in shrunk.items():
        d1 = df[(df["division"] == "D1") & x.notna()]
        avg = df["position"].map(x[d1.index].groupby(d1["position"]).mean())
        df[f"adj_{stat}"] = _discount_good_side(x, avg, trust, stat in LOWER_IS_BETTER)
    return df


# --------------------------------------------------------------------------- 3
def percentile_against(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each value within a reference sample; ties count half. NaN stays NaN."""
    ref = np.sort(reference[~np.isnan(reference)])
    vals = values.to_numpy(dtype=float)
    if len(ref) == 0:
        return np.full(len(vals), np.nan)
    left = np.searchsorted(ref, vals, side="left")
    right = np.searchsorted(ref, vals, side="right")
    return np.where(np.isnan(vals), np.nan, 100.0 * (left + right) / (2 * len(ref)))


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add pct_* = percentile of adj_* among D1 players at the same position (higher = better)."""
    df = df.copy()
    for stat in ALL_SCORED_STATS:
        df[f"pct_{stat}"] = np.nan
    for pos in POSITIONS:
        rows = df["position"] == pos
        ref = df[rows & (df["division"] == "D1")]
        for stat in ALL_SCORED_STATS:
            pct = percentile_against(df.loc[rows, f"adj_{stat}"], ref[f"adj_{stat}"].to_numpy(dtype=float))
            df.loc[rows, f"pct_{stat}"] = 100 - pct if stat in LOWER_IS_BETTER else pct
    return df


# --------------------------------------------------------------------------- 4-6
def translation_score(df: pd.DataFrame) -> pd.Series:
    """Position-weighted mean of pct_* stats (0-100), shrunk toward a prior for low minutes.

    Missing stats (no Synergy data) are skipped and the remaining weights renormalized.
    reliability = min(1, season_minutes / full_trust_minutes)
    score = reliability * weighted_pct + (1 - reliability) * prior
    """
    raw = pd.Series(np.nan, index=df.index)
    for pos, weights in POSITION_WEIGHTS.items():
        rows = df["position"] == pos
        pct = df.loc[rows, [f"pct_{s}" for s in weights]].to_numpy(dtype=float)
        w = np.array(list(weights.values()))
        have = ~np.isnan(pct)
        raw[rows] = np.nansum(pct * w, axis=1) / (have * w).sum(axis=1)
    rel = (df["season_minutes"] / SCORING_CONFIG["full_trust_minutes"]).clip(upper=1.0)
    return rel * raw + (1 - rel) * SCORING_CONFIG["reliability_prior"]


def attention_score(df: pd.DataFrame) -> pd.Series:
    """Weighted blend of percentile ranks (across all players) of attention signals.

    Followers are log-scaled first (rank is unchanged by log, but it documents
    that 10k vs 100k matters about as much as 1k vs 10k).
    """
    w = SCORING_CONFIG["attention_weights"]
    signals = {
        "recruiting_stars": df["recruiting_stars"],
        "social_followers": np.log1p(df["social_followers"]),
        "news_mentions_90d": df["news_mentions_90d"],
    }
    # Blank attention (real players without that data) is unknown, not zero: average the
    # signals we have, and leave Attention blank if none are known.
    ranks = pd.DataFrame({k: 100 * v.rank(pct=True, method="average") for k, v in signals.items()})
    weights = pd.Series(w)[ranks.columns]
    known = ranks.notna()
    return (ranks.fillna(0) * weights).sum(axis=1) / (known * weights).sum(axis=1).replace(0, np.nan)


def underrated_score(translation: pd.Series, attention: pd.Series) -> pd.Series:
    """(Translation - Attention) mapped linearly from [-100, 100] to [0, 100]. 50 = fairly rated."""
    return (translation - attention + 100) / 2


def score_players(players: pd.DataFrame, reference: pd.DataFrame | None = None,
                  baseline_label: str | None = None) -> pd.DataFrame:
    """Run the full pipeline. Returns players + derived, adj_*, pct_*, and score columns.

    reference: extra players (e.g. the synthetic D1 pool) used ONLY as the comparison
    baseline for D1 percentiles, averages, and attention ranks, then dropped. Lets a
    handful of real players be scored before a real D1 dataset exists.
    """
    df = players.assign(_reference=False)
    if reference is not None:
        ref = reference[~reference["player_id"].isin(players["player_id"])]
        df = pd.concat([df, ref.assign(_reference=True)], ignore_index=True)
    df = add_percentiles(adjust_stats(derive_stats(df)))
    df["translation_score"] = translation_score(df)
    df["attention_score"] = attention_score(df)
    df["underrated_score"] = underrated_score(df["translation_score"], df["attention_score"])
    df["baseline"] = baseline_label
    return df[~df["_reference"]].drop(columns="_reference").reset_index(drop=True)


# --------------------------------------------------------------------------- explain
def format_stat(stat: str, value: float) -> str:
    """Human-readable stat value: 41.2%, ratios 2 decimals, PPP 3 decimals, SSM signed."""
    if stat in PCT_STATS:
        return f"{100 * value:.1f}%"
    if stat in ("off_ppp", "def_ppp"):
        return f"{value:.3f}"
    if stat == "ssm":
        return f"{value:+.2f}"
    if stat == "ast_to":
        return f"{value:.2f}"
    return f"{value:.1f}"


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _stat_text(stat: str, row: pd.Series, pos: str) -> str:
    """'<label> <raw> (adjusted <adj>): <how he compares to D1 players at his position>'."""
    pct = float(row[f"pct_{stat}"])
    base = f"{STAT_LABELS[stat]} {format_stat(stat, row[stat])} (adjusted {format_stat(stat, row[f'adj_{stat}'])})"
    if stat in LOWER_IS_BETTER:  # say it in words so a high percentile can't be misread as "high turnovers"
        return f"{base}: better than {pct:.0f}% of D1 {pos}s"
    return f"{base}: {_ordinal(pct)} percentile vs D1 {pos}s"


def explain_player(row: pd.Series) -> dict[str, Any]:
    """Build the explanation object for one scored player.

    top_factors: the 3 weighted stats contributing most above average,
                 contribution = weight * (percentile - 50). Missing stats are skipped.
    top_concern: the weakest stat by percentile, including turnover rate.
    """
    pos = row["position"]
    factors = []
    for stat, w in POSITION_WEIGHTS[pos].items():
        if pd.isna(row[f"pct_{stat}"]):
            continue
        pct = float(row[f"pct_{stat}"])
        factors.append({
            "stat": stat, "label": STAT_LABELS[stat], "weight": round(w, 4),
            "raw": float(row[stat]), "adjusted": float(row[f"adj_{stat}"]), "percentile": round(pct, 1),
            "contribution": round(w * (pct - 50), 2), "text": _stat_text(stat, row, pos),
        })
    top = sorted(factors, key=lambda f: f["contribution"], reverse=True)[:3]

    # Concern: lowest percentile (all percentiles are higher = better), including turnovers per 40.
    candidates = [(f["percentile"], f["stat"]) for f in factors]
    candidates.append((float(row["pct_tov_per40"]), "tov_per40"))
    goodness, stat = min(candidates)
    raw = f"{STAT_LABELS[stat]} {format_stat(stat, row[stat])}"
    if stat in LOWER_IS_BETTER:
        text = (f"{raw} (adjusted {format_stat(stat, row[f'adj_{stat}'])}): "
                f"{_WORSE_PHRASE[stat]} {100 - goodness:.0f}% of D1 {pos}s")
    else:
        text = f"{raw}: only {_ordinal(goodness)} percentile vs D1 {pos}s"
    concern = {"stat": stat, "label": STAT_LABELS[stat], "percentile": round(goodness, 1), "text": text}

    notes = [f"Level multiplier {row['level_mult']:.2f} ({row['division']}, "
             f"conference strength {row['conference_strength']:.2f})"]
    if row["season_minutes"] < SCORING_CONFIG["full_trust_minutes"]:
        notes.append(f"Small sample: {row['season_minutes']:.0f} season minutes, "
                     f"Translation pulled toward {SCORING_CONFIG['reliability_prior']:.0f}")
    if pd.isna(row["off_ppp"]):
        notes.append("No Synergy data: scored on box-score stats only")
    if "attention_score" in row and pd.isna(row["attention_score"]):
        notes.append("No attention data (stars, followers, news): Underrated not computed")
    if isinstance(row.get("baseline"), str):
        notes.append(f"Percentiles compare him against the {row['baseline']} baseline")
    return {"top_factors": top, "top_concern": concern, "notes": notes}
