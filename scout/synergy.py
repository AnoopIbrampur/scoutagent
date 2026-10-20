"""Synergy-format situational tables: schema, synthetic generator, and identity checks.

Layout copied from Synergy's "Player Details > Overall" export (offense and defense):
one row per situation, 22 stat columns. Only the LAYOUT came from a real export;
no real numbers live in this repo.

Rules every table obeys (verified on a real export, enforced by check_identities):
  PPP  = PTS / POSS                    PPS = field-goal points / FGA (no FTs)
  SSM  = PPS - SSQ                     TSA = FGA + 0.44 * FTA,  TS% = PTS / (2 * TSA)
  %TIME = POSS / parent POSS           FG MISS = FG ATT - FG MADE
  Transition + Overall Half Court = All Possessions
  Against Man + Against Zone = Overall Half Court
  To Transition + To Half Court = Press
  OOB End/Side, After Time Outs, Last 4 Sec. are overlapping subsets of Half Court.

Synthetic generation works on integer COUNTS (2PA/2PM, 3PA/3PM, FTA/FTM, TOs,
free-throw trips, scoring possessions, expected shot points) so the rules hold
exactly; display columns are computed from counts at the end. The All
Possessions offense row is built from the player's box-score season totals, so
Synergy and box score agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# (snake_case key, Synergy column label) in export order.
SYNERGY_COLUMNS: list[tuple[str, str]] = [
    ("poss", "POSS"), ("pct_time", "%TIME"), ("pct_time_rank", "%TIME RANK"), ("pts", "PTS"),
    ("ppp", "PPP"), ("ppp_rank", "PPP RANK"), ("rating", "RATING"), ("fg_att", "FG ATT"),
    ("fg_made", "FG MADE"), ("fg_miss", "FG MISS"), ("fg_pct", "FG%"), ("efg_pct", "EFG%"),
    ("ssq", "SSQ"), ("pps", "PPS"), ("ssm", "SSM"), ("ts_pct", "TS%"), ("tsa", "TSA"),
    ("to_pct", "TO%"), ("ft_freq", "%FT"), ("fta_per_fga", "FTA/FGA"), ("sf_freq", "%SF"),
    ("score_pct", "SCORE%"),
]
STAT_KEYS = [k for k, _ in SYNERGY_COLUMNS]
PERCENT_KEYS = {"pct_time", "fg_pct", "efg_pct", "ts_pct", "to_pct", "ft_freq", "sf_freq", "score_pct"}
RANK_KEYS = {"pct_time_rank", "ppp_rank"}

OFFENSE_SITUATIONS = [
    "All Possessions", "Transition", "Overall Half Court", "Out of Bounds (End)", "Out of Bounds (Side)",
    "After Time Outs", "Last 4 Sec. of Shot Clock", "Against Man", "Against Zone", "Press",
    "To Transition", "To Half Court",
]
# Defense uses the same rows, except man/zone are labelled from the defender's side.
DEFENSE_SITUATIONS = [s.replace("Against Man", "Man Defense").replace("Against Zone", "Zone Defense")
                      for s in OFFENSE_SITUATIONS]
ALL_SITUATIONS = set(OFFENSE_SITUATIONS) | set(DEFENSE_SITUATIONS)

PARENT = {
    "Transition": "All Possessions", "Overall Half Court": "All Possessions", "Press": "All Possessions",
    "Out of Bounds (End)": "Overall Half Court", "Out of Bounds (Side)": "Overall Half Court",
    "After Time Outs": "Overall Half Court", "Last 4 Sec. of Shot Clock": "Overall Half Court",
    "Against Man": "Overall Half Court", "Against Zone": "Overall Half Court",
    "Man Defense": "Overall Half Court", "Zone Defense": "Overall Half Court",
    "To Transition": "Press", "To Half Court": "Press",
}
PARTITIONS = [  # (parent, child_a, child_b): child_a + child_b == parent
    ("All Possessions", "Transition", "Overall Half Court"),
    ("Overall Half Court", "Against Man", "Against Zone"),
    ("Overall Half Court", "Man Defense", "Zone Defense"),
    ("Press", "To Transition", "To Half Court"),
]
MIN_POSS_FOR_RANK = 10  # Synergy shows "-" for ranks on tiny samples


def rating_for(rank: float) -> str | None:
    """Synergy-style label from a 0-100 PPP rank. Thresholds approximated from real exports."""
    if pd.isna(rank):
        return None
    for floor, label in ((85, "Excellent"), (65, "Very Good"), (45, "Good"), (30, "Average"), (10, "Below Average")):
        if rank >= floor:
            return label
    return "Poor"


# --------------------------------------------------------------------------- counts
COUNT_KEYS = ["two_a", "two_m", "three_a", "three_m", "fta", "ftm", "tov", "trips", "sf", "fto", "score"]


def _poss(c: dict[str, float]) -> int:
    """Possessions = shots + turnovers + free-throw-only trips."""
    return int(c["two_a"] + c["three_a"] + c["tov"] + c["fto"])


def _fgm(c: dict[str, float]) -> int:
    return int(c["two_m"] + c["three_m"])


def _alloc_made(made: int, att: int, att_child: int, rate_mult: float, rng: np.random.Generator) -> int:
    """Makes for a child that took `att_child` of the parent's `att` attempts.

    Keeps both children feasible: the child gets at most att_child makes and the
    remainder never has more makes than attempts.
    """
    lo, hi = max(0, made - (att - att_child)), min(att_child, made)
    if att == 0 or hi <= lo:
        return lo
    expected = att_child * (made / att) * rate_mult + rng.normal(0, 0.5)
    return int(np.clip(round(expected), lo, hi))


def _split(parent: dict[str, float], share: float, eff: float, ssq_mult: float,
           rng: np.random.Generator) -> dict[str, float]:
    """Draw a child situation holding roughly `share` of the parent's possessions.

    eff tilts shooting efficiency (transition > half court > end of clock);
    ssq_mult tilts expected shot quality the same way.
    """
    c: dict[str, float] = {}
    for a, m in (("two_a", "two_m"), ("three_a", "three_m")):
        c[a] = int(rng.binomial(int(parent[a]), share))
        c[m] = _alloc_made(int(parent[m]), int(parent[a]), int(c[a]), eff, rng)
    c["tov"] = int(rng.binomial(int(parent["tov"]), share))
    c["trips"] = int(rng.binomial(int(parent["trips"]), share))
    for k in ("sf", "fto"):  # both are subsets of trips
        c[k] = _alloc_made(int(parent[k]), int(parent["trips"]), c["trips"], 1.0, rng)
    # FTA: at least one per trip in both children.
    ratio = parent["fta"] / parent["trips"] if parent["trips"] else 0
    lo, hi = c["trips"], int(parent["fta"]) - (int(parent["trips"]) - c["trips"])
    c["fta"] = int(np.clip(round(c["trips"] * ratio), lo, max(lo, hi)))
    c["ftm"] = _alloc_made(int(parent["ftm"]), int(parent["fta"]), c["fta"], 1.0, rng)
    # Scoring possessions: every made FG scores, and never more than possessions.
    poss_c, fgm_c = _poss(c), _fgm(c)
    poss_r, fgm_r = _poss(parent) - poss_c, _fgm(parent) - fgm_c
    lo = max(fgm_c, int(parent["score"]) - poss_r)
    hi = max(lo, min(poss_c, int(parent["score"]) - fgm_r))
    exp = poss_c * (parent["score"] / max(_poss(parent), 1)) * eff
    c["score"] = int(np.clip(round(exp), lo, hi))
    # Expected shot points (SSQ * FGA). Child gets a tilted per-shot quality.
    p_fga = parent["two_a"] + parent["three_a"]
    per_shot = parent["ssq_pts"] / p_fga if p_fga else 0.0
    c_fga = c["two_a"] + c["three_a"]
    c["ssq_pts"] = min(parent["ssq_pts"], c_fga * per_shot * ssq_mult)
    return c


def _minus(parent: dict[str, float], child: dict[str, float]) -> dict[str, float]:
    return {k: parent[k] - child[k] for k in parent}


def _build_situations(all_row: dict[str, float], defense: bool, guard_like: float,
                      rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """All 12 situations as count dicts, derived from the All Possessions counts."""
    rows: dict[str, dict[str, float]] = {"All Possessions": all_row}
    t_share = 0.0 if defense else float(np.clip(0.08 + 0.10 * guard_like + rng.normal(0, 0.02), 0.03, 0.25))
    rows["Transition"] = _split(all_row, t_share, 1.12, 1.12, rng)
    hc = rows["Overall Half Court"] = _minus(all_row, rows["Transition"])
    man, zone = ("Man Defense", "Zone Defense") if defense else ("Against Man", "Against Zone")
    rows[zone] = _split(hc, rng.uniform(0.02, 0.06) if defense else rng.uniform(0.08, 0.18), 1.03, 1.0, rng)
    rows[man] = _minus(hc, rows[zone])
    for name, lo, hi, eff in (("Out of Bounds (End)", 0.03, 0.08, 1.0), ("Out of Bounds (Side)", 0.02, 0.05, 1.0),
                              ("After Time Outs", 0.12, 0.24, 0.98), ("Last 4 Sec. of Shot Clock", 0.04, 0.08, 0.85)):
        rows[name] = _split(hc, rng.uniform(lo, hi), eff, eff, rng)
    rows["Press"] = _split(all_row, rng.uniform(0.01, 0.03) if defense else rng.uniform(0.03, 0.08), 0.97, 0.98, rng)
    rows["To Transition"] = _split(rows["Press"], 0.0 if defense else rng.uniform(0.1, 0.3), 1.1, 1.1, rng)
    rows["To Half Court"] = _minus(rows["Press"], rows["To Transition"])
    return rows


def _metrics(c: dict[str, float], parent_poss: int | None) -> dict[str, Any]:
    """Synergy display columns from counts ('-' in the export becomes NaN here)."""
    nan = float("nan")
    poss, fgm = _poss(c), _fgm(c)
    fga = int(c["two_a"] + c["three_a"])
    fg_pts = 2 * c["two_m"] + 3 * c["three_m"]
    pts = int(fg_pts + c["ftm"])
    tsa = fga + 0.44 * c["fta"]
    pps = round(fg_pts / fga, 2) if fga else nan
    ssq = round(c["ssq_pts"] / fga, 2) if fga else nan
    return {
        "poss": poss, "pct_time": (poss / parent_poss if parent_poss else (1.0 if parent_poss is None else 0.0)),
        "pts": pts, "ppp": round(pts / poss, 3) if poss else nan,
        "fg_att": fga, "fg_made": fgm, "fg_miss": fga - fgm,
        "fg_pct": fgm / fga if fga else nan, "efg_pct": (fgm + 0.5 * c["three_m"]) / fga if fga else nan,
        "ssq": ssq, "pps": pps, "ssm": round(pps - ssq, 2) if fga else nan,
        "ts_pct": pts / (2 * tsa) if tsa else nan, "tsa": int(round(tsa)),
        "to_pct": c["tov"] / poss if poss else nan, "ft_freq": c["trips"] / poss if poss else nan,
        "fta_per_fga": c["fta"] / fga if fga else nan, "sf_freq": c["sf"] / poss if poss else nan,
        "score_pct": c["score"] / poss if poss else nan,
    }


# --------------------------------------------------------------------------- player-level inputs
POSITION_SSQ = {"PG": 0.97, "SG": 0.99, "SF": 1.00, "PF": 1.04, "C": 1.10}  # expected points per shot
GUARD_LIKE = {"PG": 1.0, "SG": 0.9, "SF": 0.6, "PF": 0.3, "C": 0.1}


def _offense_counts(p: pd.Series, fga_pg: float, rng: np.random.Generator) -> dict[str, float]:
    """Season counts from the box score, so Synergy agrees with per-game stats."""
    g = int(p["games"])
    fga = int(round(fga_pg * g))
    three_a = min(int(round(p["three_att_pg"] * g)), fga)
    three_m = int(round(p["three_pct"] * three_a))
    fgm = int(np.clip(round(p["fg_pct"] * fga), three_m, three_m + fga - three_a))
    fta = int(round(p["ft_att_pg"] * g))
    trips = int(np.clip(round(fta / 1.75), int(np.ceil(fta / 3)), fta))
    c = {
        "two_a": fga - three_a, "two_m": fgm - three_m, "three_a": three_a, "three_m": three_m,
        "fta": fta, "ftm": int(round(p["ft_pct"] * fta)), "tov": int(round(p["turnovers_pg"] * g)),
        "trips": trips, "sf": int(round(trips * rng.uniform(0.75, 0.85))),
        "fto": int(round(trips * rng.uniform(0.8, 0.9))),
    }
    poss = _poss(c)
    ft_score = 1 - (1 - p["ft_pct"]) ** 2  # chance a 2-shot trip scores at least once
    c["score"] = int(np.clip(round(fgm + c["fto"] * ft_score), fgm, poss))
    ssq = POSITION_SSQ[p["position"]] + 0.08 * (three_a / fga if fga else 0) + rng.normal(0, 0.04)
    c["ssq_pts"] = ssq * fga
    return c


def _defense_counts(p: pd.Series, d: float, rng: np.random.Generator) -> dict[str, float]:
    """Opponent results on possessions this player defended. d > 0 means a better defender."""
    poss = int(round(p["games"] * p["minutes_pg"] / 32 * rng.uniform(5.0, 9.0)))
    tov = int(rng.binomial(poss, float(np.clip(0.03 + 0.01 * d, 0.005, 0.08))))
    trips = int(rng.binomial(poss, float(np.clip(0.06 - 0.01 * d, 0.01, 0.12))))
    fto = int(round(trips * rng.uniform(0.85, 0.95)))
    fga = max(poss - tov - fto, 0)
    three_a = int(rng.binomial(fga, float(np.clip(0.35 + 0.15 * (GUARD_LIKE[p["position"]] - 0.5), 0.15, 0.55))))
    two_a = fga - three_a
    two_m = int(rng.binomial(two_a, float(np.clip(0.47 - 0.025 * d + rng.normal(0, 0.02), 0.3, 0.65))))
    three_m = int(rng.binomial(three_a, float(np.clip(0.33 - 0.015 * d + rng.normal(0, 0.02), 0.2, 0.45))))
    fta = int(round(trips * 1.9))
    c = {"two_a": two_a, "two_m": two_m, "three_a": three_a, "three_m": three_m, "fta": fta,
         "ftm": int(rng.binomial(fta, 0.7)), "tov": tov, "trips": trips,
         "sf": int(round(trips * 0.9)), "fto": fto}
    c["score"] = int(np.clip(round(two_m + three_m + fto * 0.85), two_m + three_m, _poss(c)))
    c["ssq_pts"] = (0.92 - 0.02 * d + rng.normal(0, 0.03)) * fga
    return c


def _table(player_id: str, rows: dict[str, dict[str, float]], situations: list[str]) -> list[dict[str, Any]]:
    out = []
    for name in situations:
        parent = PARENT.get(name)
        parent_poss = _poss(rows[parent]) if parent else None
        out.append({"player_id": player_id, "situation": name, **_metrics(rows[name], parent_poss)})
    return out


def add_ranks(table: pd.DataFrame, defense: bool) -> pd.DataFrame:
    """Percentile ranks (1-100) within each situation, plus RATING. Defense: lower PPP ranks higher."""
    table = table.copy()
    table["pct_time_rank"] = np.nan
    table["ppp_rank"] = np.nan
    for _, idx in table.groupby("situation").groups.items():
        sub = table.loc[idx]
        ok = sub["poss"] >= MIN_POSS_FOR_RANK
        if ok.sum() < 2:
            continue
        ppp = sub.loc[ok, "ppp"]
        table.loc[ppp.index, "ppp_rank"] = (100 * (-ppp if defense else ppp).rank(pct=True)).round().clip(1, 100)
        if sub["situation"].iloc[0] != "All Possessions":
            t = sub.loc[ok, "pct_time"]
            table.loc[t.index, "pct_time_rank"] = (100 * t.rank(pct=True)).round().clip(1, 100)
    table["rating"] = table["ppp_rank"].map(rating_for)
    return table[["player_id", "situation"] + STAT_KEYS]


def generate_synergy(players: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(offense, defense) tables, 12 situations per player, derived from box scores."""
    from scout.scoring import estimate_fga  # local import: scoring imports nothing from here

    rng = np.random.default_rng(seed + 1)
    fga_pg = estimate_fga(players)
    # Defensive skill proxy: steals + blocks per 40, relative to position, plus noise.
    per40 = (players["steals_pg"] + players["blocks_pg"]) * 40 / players["minutes_pg"].clip(lower=1)
    z = per40.groupby(players["position"]).transform(lambda s: (s - s.mean()) / (s.std() or 1))
    d_skill = 0.6 * z + rng.normal(0, 0.8, len(players))

    off_rows, def_rows = [], []
    for i, p in players.iterrows():
        off = _build_situations(_offense_counts(p, float(fga_pg[i]), rng), False, GUARD_LIKE[p["position"]], rng)
        dfn = _build_situations(_defense_counts(p, float(d_skill[i]), rng), True, 0.0, rng)
        off_rows += _table(p["player_id"], off, OFFENSE_SITUATIONS)
        def_rows += _table(p["player_id"], dfn, DEFENSE_SITUATIONS)
    return add_ranks(pd.DataFrame(off_rows), False), add_ranks(pd.DataFrame(def_rows), True)


# --------------------------------------------------------------------------- validation
def check_identities(table: pd.DataFrame) -> list[str]:
    """Return rule violations (empty list = consistent). Works on synthetic or parsed real tables.

    Tolerances allow for Synergy's display rounding.
    """
    errors = []
    for pid, t in table.groupby("player_id"):
        rows = t.set_index("situation")
        for name, r in rows.iterrows():
            tag = f"{pid}/{name}"
            if r["poss"] and not pd.isna(r["ppp"]) and abs(r["pts"] / r["poss"] - r["ppp"]) > 0.0015:
                errors.append(f"{tag}: PPP != PTS/POSS")
            if r["fg_att"] - r["fg_made"] != r["fg_miss"]:
                errors.append(f"{tag}: FG MISS != ATT - MADE")
            if r["fg_att"] and abs(r["fg_made"] / r["fg_att"] - r["fg_pct"]) > 0.0006:
                errors.append(f"{tag}: FG% != MADE/ATT")
            if not pd.isna(r["ssm"]) and abs(r["pps"] - r["ssq"] - r["ssm"]) > 0.011:
                errors.append(f"{tag}: SSM != PPS - SSQ")
            if r["tsa"] and not pd.isna(r["ts_pct"]):
                # TSA is displayed rounded to a whole number, so allow +/- 0.5 on it.
                lo, hi = r["pts"] / (2 * (r["tsa"] + 0.5)), r["pts"] / (2 * max(r["tsa"] - 0.5, 0.25))
                if not lo - 0.0006 <= r["ts_pct"] <= hi + 0.0006:
                    errors.append(f"{tag}: TS% != PTS / (2 * TSA)")
            parent = PARENT.get(name)
            if parent in rows.index and rows.loc[parent, "poss"]:
                if abs(r["poss"] / rows.loc[parent, "poss"] - r["pct_time"]) > 0.0015:
                    errors.append(f"{tag}: %TIME != POSS / parent POSS")
        for parent, a, b in PARTITIONS:
            if {parent, a, b} <= set(rows.index):
                for col in ("poss", "pts", "fg_att", "fg_made"):
                    if rows.loc[a, col] + rows.loc[b, col] != rows.loc[parent, col]:
                        errors.append(f"{pid}: {a} + {b} != {parent} ({col})")
    return errors


# --------------------------------------------------------------------------- scoring hook
def overall_rows(offense: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the All Possessions numbers scoring uses, with prefixed names."""
    o = offense[offense["situation"] == "All Possessions"].set_index("player_id")
    d = defense[defense["situation"] == "All Possessions"].set_index("player_id")
    out = pd.DataFrame({
        "syn_off_poss": o["poss"], "off_ppp": o["ppp"], "off_ppp_rank": o["ppp_rank"],
        "off_rating": o["rating"], "ssq": o["ssq"], "ssm": o["ssm"], "syn_fga": o["fg_att"],
        "to_pct": o["to_pct"],
    }).join(pd.DataFrame({
        "syn_def_poss": d["poss"], "def_ppp": d["ppp"], "def_ppp_rank": d["ppp_rank"], "def_rating": d["rating"],
    }), how="outer")
    return out.reset_index()


def attach_synergy(players: pd.DataFrame, offense: pd.DataFrame | None,
                   defense: pd.DataFrame | None) -> pd.DataFrame:
    """Players plus Synergy overall columns (NaN for players without Synergy data)."""
    if offense is None or defense is None or offense.empty:
        return players
    return players.merge(overall_rows(offense, defense), on="player_id", how="left")
