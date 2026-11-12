"""The agency view of a player: should we represent him, where should he go, what do we lead with.

Three pieces, all built on the scored players table and the Synergy play-type tables:

  representation_call  Sign / Watch / Pass, from Translation and Underrated (thresholds in config).
  best_fits            D1 programs ranked by need at his position (60%) and style match (40%).
                         need  = 0.5 * opening + 0.5 * edge
                           opening: share of the program's minutes at his position held by seniors
                           edge:    1 if he projects above their best returning player at his
                                    position (by Translation), fading to 0 at 20 points below
                         style = how closely the program's play-type mix matches his, scaled
                                 0-1 across the programs considered
  leverage_points      The numbers to lead with when talking to a program or a collective.

All of it is computed from the data in the app. The programs in the hosted demo are synthetic,
so their rosters and styles are invented; with real rosters and team Synergy exports the same
code ranks real destinations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scout.config import REPRESENTATION_RULES
from scout.scoring import STAT_LABELS, format_stat

NEED_WEIGHT, STYLE_WEIGHT = 0.6, 0.4
EDGE_FADE = 20.0  # Translation points below the best returning incumbent at which "edge" reaches 0
LEVERAGE_MIN_PCT = 85.0
MARKETABILITY_LEVERAGE = 60.0


# --------------------------------------------------------------------------- representation call
def representation_call(translation: float, underrated: float | None) -> tuple[str, str]:
    """('Sign' | 'Watch' | 'Pass', one-line reason). Unknown Underrated doesn't block 'Sign'."""
    sign, watch = REPRESENTATION_RULES["sign"], REPRESENTATION_RULES["watch"]
    if translation >= sign["min_translation"] and (underrated is None or underrated >= sign["min_underrated"]):
        why = "projects as a D1 contributor"
        why += " and the market hasn't priced him in" if underrated is not None else ""
        return "Sign", why[0].upper() + why[1:] + "."
    if translation >= watch["min_translation"]:
        return "Watch", "D1-viable but not a lock; track him against his best opponents."
    return "Pass", "Doesn't project to a D1 rotation role yet."


# --------------------------------------------------------------------------- best-fit destinations
def program_styles(scored: pd.DataFrame, playtypes_offense: pd.DataFrame) -> pd.DataFrame:
    """Share of each D1 program's offensive possessions by play type (rows = schools)."""
    d1 = scored.loc[scored["division"] == "D1", ["player_id", "school"]]
    pt = playtypes_offense.merge(d1, on="player_id")
    table = pt.pivot_table(index="school", columns="situation", values="poss", aggfunc="sum").fillna(0)
    return table.div(table.sum(axis=1), axis=0)


def _player_style(playtypes_offense: pd.DataFrame, player_id: str) -> pd.Series | None:
    rows = playtypes_offense[playtypes_offense["player_id"] == player_id].set_index("situation")["poss"]
    return rows / rows.sum() if len(rows) and rows.sum() > 0 else None


def best_fits(row: pd.Series, scored: pd.DataFrame, playtypes_offense: pd.DataFrame | None,
              styles: pd.DataFrame | None = None, n: int = 5) -> list[dict[str, Any]]:
    """Top-n D1 programs for this player, with the numbers behind each and a one-line reason."""
    pos, pid, me = row["position"], row["player_id"], float(row["translation_score"])
    d1 = scored[(scored["division"] == "D1") & (scored["school"] != row["school"])]
    mine = _player_style(playtypes_offense, pid) if playtypes_offense is not None else None
    if mine is not None and styles is None:
        styles = program_styles(scored, playtypes_offense)
    top_type = mine.idxmax() if mine is not None else None

    out = []
    for school, roster in d1.groupby("school"):
        at_pos = roster[roster["position"] == pos]
        if at_pos.empty:
            # No real D1 program has no one at a position; in the synthetic data this only means
            # positions were dealt out at random, so it isn't a need signal. Skip it.
            continue
        minutes = float(at_pos["minutes_pg"].sum())
        seniors = at_pos[at_pos["class_year"] == "SR"]
        opening = float(seniors["minutes_pg"].sum()) / minutes if minutes else 1.0
        returning = at_pos[at_pos["class_year"] != "SR"]
        best_return = float(returning["translation_score"].max()) if len(returning) else None
        edge = 1.0 if best_return is None else float(np.clip(1 - (best_return - me) / EDGE_FADE, 0, 1))
        out.append({"school": school, "conference": roster["conference"].iloc[0], "at_position": int(len(at_pos)),
                    "seniors_leaving": int(len(seniors)), "opening_share": round(opening, 3),
                    "best_returning_translation": None if best_return is None else round(best_return, 1),
                    "need": 0.5 * opening + 0.5 * edge, "style_raw": np.nan})
        if mine is not None and school in styles.index:
            team = styles.loc[school].reindex(mine.index).fillna(0)
            out[-1]["style_raw"] = float(np.dot(team, mine) / (np.linalg.norm(team) * np.linalg.norm(mine)))
            out[-1]["program_top_share"] = round(float(team[top_type]), 3)

    fits = pd.DataFrame(out)
    if fits["style_raw"].notna().any():
        lo, hi = fits["style_raw"].min(), fits["style_raw"].max()
        fits["style"] = (fits["style_raw"] - lo) / (hi - lo) if hi > lo else 1.0
        fits["fit"] = NEED_WEIGHT * fits["need"] + STYLE_WEIGHT * fits["style"]
    else:  # no play-type data: rank on need alone
        fits["style"], fits["fit"] = np.nan, fits["need"]
    fits = fits.sort_values(["fit", "need"], ascending=False).head(n)

    result = []
    for _, f in fits.iterrows():
        result.append({
            "school": f["school"], "conference": f["conference"], "fit": round(100 * f["fit"], 1),
            "need": round(100 * f["need"], 1),
            "style": None if pd.isna(f["style"]) else round(100 * f["style"], 1),
            "seniors_leaving": f["seniors_leaving"], "opening_share": f["opening_share"],
            "best_returning_translation": None if pd.isna(f["best_returning_translation"])
            else f["best_returning_translation"],
            "reason": _fit_reason(f, pos, me, top_type),
        })
    return result


def _fit_reason(f: pd.Series, pos: str, me: float, top_type: str | None) -> str:
    """One plain sentence built only from numbers in the fit row (so the fact-check can verify it)."""
    parts = []
    if f["seniors_leaving"]:
        s = "s" if f["seniors_leaving"] != 1 else ""
        parts.append(f"{f['seniors_leaving']} senior{s} leaving at {pos} ({100 * f['opening_share']:.0f}% of {pos} minutes)")
    if f["at_position"] and pd.isna(f["best_returning_translation"]):
        parts.append(f"no returning {pos}")
    elif not pd.isna(f["best_returning_translation"]) and me >= f["best_returning_translation"]:
        parts.append(f"projects above their best returning {pos} (Translation {f['best_returning_translation']:.1f})")
    if top_type is not None and not pd.isna(f.get("program_top_share", np.nan)):
        parts.append(f"runs {top_type} on {100 * f['program_top_share']:.1f}% of possessions, "
                     f"his most-used play type")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "." if text else "Fits on overall need and style."


# --------------------------------------------------------------------------- leverage points
def leverage_points(row: pd.Series, explanation: dict[str, Any], n: int = 3) -> list[str]:
    """The strongest selling points: his best D1-percentile stats, plus marketability if high."""
    pos = row["position"]
    factors = sorted([f for f in explanation["top_factors"] if f["percentile"] >= LEVERAGE_MIN_PCT],
                     key=lambda f: f["percentile"], reverse=True)
    points = [f"{STAT_LABELS[f['stat']]} of {format_stat(f['stat'], f['raw'])}: better than "
              f"{min(f['percentile'], 99.0):.0f}% of D1 {pos}s after the level adjustment" for f in factors[:n]]
    mkt = row.get("marketability_score", np.nan)
    if not pd.isna(mkt) and mkt >= MARKETABILITY_LEVERAGE:
        points.append(f"Marketability {mkt:.1f}: more followers than {mkt:.0f}% of players in the database, "
                      "an NIL selling point")
    return points
