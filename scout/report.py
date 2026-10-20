"""Scouting report generation with a hallucination guard.

Flow:
  build_report_input()  -> the ONLY facts the writer may use (formatted, rounded)
  generate_report()     -> cache hit? return it.
                           else Claude (if key) -> number check -> one retry if flagged
                           else template report (offline, deterministic)
  check_numbers()       -> every number in the report must trace back to the input

Reports are cached in DuckDB (table `reports`) keyed by player_id plus a hash
of the input, so the demo never re-calls the API, and a rescored player
automatically gets a fresh report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from scout.config import DB_PATH, SCOUT_MODEL, has_api_key
from scout.db import get_connection
from scout.llm import LLMError, complete
from scout.query_parser import inches_to_str
from scout.scoring import STAT_LABELS, explain_player

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"  # bump to invalidate cached reports after prompt changes
SECTIONS = [
    "Snapshot", "Strengths", "Weaknesses / Questions", "Projection at D1 Level",
    "Comparable Player Archetype", "Recommendation",
]
RECOMMENDATIONS = ("Pursue", "Monitor", "Pass")


class ReportResult(BaseModel):
    """A generated (or cached) report plus its validation results."""

    player_id: str
    markdown: str
    method: Literal["llm", "template"]
    model: str | None = None
    cached: bool = False
    flagged_numbers: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- input facts
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def suggested_recommendation(translation: float, underrated: float) -> str:
    """Rule of thumb shared by the template and the LLM prompt."""
    if translation >= 75 and underrated >= 65:
        return "Pursue"
    if translation >= 55:
        return "Monitor"
    return "Pass"


def build_report_input(row: pd.Series) -> dict[str, Any]:
    """Everything the report may cite, already rounded the way it should be quoted."""
    exp = explain_player(row)
    weighted = [f["stat"] for f in exp["top_factors"]] + [exp["top_concern"]["stat"]]
    pct_stats = list(dict.fromkeys(weighted + ["ts_pct", "ast_to", "pts_per40", "reb_per40", "ast_per40", "tov_per40"]))
    t, u = round(float(row["translation_score"]), 1), round(float(row["underrated_score"]), 1)
    return {
        "player": {
            "name": row["name"], "school": row["school"], "conference": row["conference"],
            "division": row["division"], "position": row["position"],
            "height": inches_to_str(int(row["height_in"])), "weight_lb": int(row["weight_lb"]),
            "class_year": row["class_year"],
        },
        "season_per_game": {
            "games": int(row["games"]), "minutes": round(float(row["minutes_pg"]), 1),
            "points": round(float(row["points_pg"]), 1), "rebounds": round(float(row["rebounds_pg"]), 1),
            "assists": round(float(row["assists_pg"]), 1), "steals": round(float(row["steals_pg"]), 1),
            "blocks": round(float(row["blocks_pg"]), 1), "turnovers": round(float(row["turnovers_pg"]), 1),
            "fg_pct": _pct(row["fg_pct"]), "three_pct": _pct(row["three_pct"]),
            "three_attempts": round(float(row["three_att_pg"]), 1),
            "ft_pct": _pct(row["ft_pct"]), "ft_attempts": round(float(row["ft_att_pg"]), 1),
        },
        "derived": {
            "true_shooting_pct": _pct(row["ts_pct"]), "assist_to_turnover": round(float(row["ast_to"]), 2),
            "points_per40": round(float(row["pts_per40"]), 1), "rebounds_per40": round(float(row["reb_per40"]), 1),
            "assists_per40": round(float(row["ast_per40"]), 1), "turnovers_per40": round(float(row["tov_per40"]), 1),
            "season_minutes": int(round(float(row["season_minutes"]))),
        },
        "d1_adjusted": {
            "level_multiplier": round(float(row["level_mult"]), 2),
            "conference_strength": round(float(row["conference_strength"]), 2),
            "points_per40": round(float(row["adj_pts_per40"]), 1),
            "rebounds_per40": round(float(row["adj_reb_per40"]), 1),
            "assists_per40": round(float(row["adj_ast_per40"]), 1),
            "turnovers_per40": round(float(row["adj_tov_per40"]), 1),
            "true_shooting_pct": _pct(row["adj_ts_pct"]),
        },
        "d1_position_percentiles": {STAT_LABELS[s]: int(round(float(row[f"pct_{s}"]))) for s in pct_stats},
        "attention": {
            "recruiting_stars": int(row["recruiting_stars"]),
            "social_followers": int(row["social_followers"]),
            "news_mentions_90d": int(row["news_mentions_90d"]),
        },
        "scores": {"translation": t, "attention": round(float(row["attention_score"]), 1), "underrated": u},
        "explanation": {
            "top_factors": [f["text"] for f in exp["top_factors"]],
            "top_concern": exp["top_concern"]["text"],
            "notes": exp["notes"],
        },
        "suggested_recommendation": suggested_recommendation(t, u),
    }


def input_hash(facts: dict[str, Any]) -> str:
    """Stable hash of the facts + prompt version (cache key)."""
    blob = json.dumps(facts, sort_keys=True) + PROMPT_VERSION
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- number check
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_HEIGHT_TOKEN = re.compile(r"\b[5-7]\s*['’]\s*\d{1,2}\s*(?:\"|”|'')?")
# Structural mentions that contain digits but are not stats.
_STRUCTURAL = [
    r"\bD-?[123]\b", r"\bDivision\s+(?:I{1,3}|[123])\b", r"\bper[- ]?40\b", r"\b40[- ]minutes?\b",
    r"\b90[- ]days?\b", r"\b3[- ]and[- ]D\b", r"\b[23][- ]level\b", r"\b1[- ]on[- ]1\b", r"\b3P%?", r"\b3PA\b", r"\b3PT\b", r"\b3-point\w*", r"\bthree-point\w*", r"\b0\s*[-–]\s*100\b",
]


def _collect_numbers(obj: Any, out: set[float]) -> None:
    """Gather every number in the facts, plus the forms a writer might quote."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            out.add(float(m.group().replace(",", "")))


def allowed_numbers(facts: dict[str, Any]) -> set[float]:
    """Numbers in the input, plus percentile complements (99th -> "top 1%") and 100."""
    nums: set[float] = set()
    _collect_numbers(facts, nums)
    for v in facts["d1_position_percentiles"].values():
        nums.add(100.0 - v)
    nums.update({40.0, 100.0})
    return nums


def _matches(token: str, allowed: set[float]) -> bool:
    """True if `token` equals some allowed number at the token's own precision."""
    x = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(round(a, decimals) == x for a in allowed)


def check_numbers(markdown: str, facts: dict[str, Any]) -> list[str]:
    """Return numbers in the report that are not supported by the input facts."""
    text = markdown
    flagged = []
    for m in _HEIGHT_TOKEN.finditer(text):  # heights must equal the player's height
        h = re.sub(r"\s", "", m.group()).replace("’", "'").replace("”", '"').replace("''", '"')
        if h.rstrip('"') != facts["player"]["height"].rstrip('"'):
            flagged.append(m.group().strip())
    text = _HEIGHT_TOKEN.sub(" ", text)
    for pat in _STRUCTURAL:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    allowed = allowed_numbers(facts)
    for m in _NUM.finditer(text):
        if not _matches(m.group(), allowed):
            flagged.append(m.group())
    return list(dict.fromkeys(flagged))


def missing_sections(markdown: str) -> list[str]:
    """Required '## Section' headers that are absent."""
    present = {h.strip().lower() for h in re.findall(r"^#{2,3}\s*(.+?)\s*$", markdown, flags=re.MULTILINE)}
    return [s for s in SECTIONS if s.lower() not in present]


# --------------------------------------------------------------------------- writers
_ARCHETYPES = {
    ("PG", "passing"): "Pass-first floor general who wins with decisions and tempo rather than burst",
    ("PG", "shooting"): "Shooting point guard who punishes drop coverage and can play off the ball",
    ("PG", "other"): "Steady lead guard who keeps the offense organized",
    ("SG", "shooting"): "Movement shooter who spaces the floor and scores efficiently off the catch",
    ("SG", "other"): "Scoring guard who creates his own looks",
    ("SF", "shooting"): "3-and-D style wing whose value starts with efficient shooting",
    ("SF", "other"): "Versatile wing who contributes across the box score",
    ("PF", "other"): "Efficient frontcourt finisher who rebounds his position",
    ("C", "other"): "Rim-running center who finishes efficiently and controls the glass",
}
_SKILL_OF_STAT = {"ast_to": "passing", "ast_per40": "passing", "three_pct_adj": "shooting", "ts_pct": "shooting"}


def template_report(facts: dict[str, Any], row: pd.Series) -> str:
    """Deterministic offline report. Uses only values from `facts`, so it always passes the check."""
    p, s, d, a, sc, e = (facts[k] for k in ("player", "season_per_game", "derived", "d1_adjusted", "scores", "explanation"))
    att = facts["attention"]
    rec = facts["suggested_recommendation"]
    exp = explain_player(row)
    skill = _SKILL_OF_STAT.get(exp["top_factors"][0]["stat"], "other")
    archetype = _ARCHETYPES.get((p["position"], skill), _ARCHETYPES[(p["position"], "other")])
    reason = {
        "Pursue": f"Translation {sc['translation']} with Underrated {sc['underrated']}: D1-level production the market has not priced in.",
        "Monitor": f"Translation {sc['translation']} is D1-viable but not a lock; see him against his best opponents.",
        "Pass": f"Translation {sc['translation']} does not project to a D1 rotation role yet.",
    }[rec]
    weaknesses = [f"- {e['top_concern']}"] + [f"- {n}" for n in e["notes"] if n.startswith("Small sample")]
    stars = "no recruiting stars" if att["recruiting_stars"] == 0 else f"a {att['recruiting_stars']}-star rating"
    news = "no news mentions" if att["news_mentions_90d"] == 0 else f"{att['news_mentions_90d']} news mentions"
    weaknesses.append(f"- Question: {stars} and {news} in 90 days, so little outside film or "
                      f"evaluation exists. Verify the numbers in person.")
    return "\n".join([
        "## Snapshot",
        f"{p['class_year']} {p['height']} {p['position']} at {p['school']} ({p['division']}, {p['conference']}): "
        f"{s['points']} PPG, {s['rebounds']} RPG, {s['assists']} APG on {d['true_shooting_pct']} true shooting. "
        f"Translation {sc['translation']}, Underrated {sc['underrated']}.",
        "",
        "## Strengths",
        *[f"- {t}" for t in e["top_factors"]],
        "",
        "## Weaknesses / Questions",
        *weaknesses,
        "",
        "## Projection at D1 Level",
        f"After the {a['level_multiplier']} level adjustment, his production projects to "
        f"{a['points_per40']} points, {a['rebounds_per40']} rebounds and {a['assists_per40']} assists per 40 "
        f"minutes against D1 competition, with true shooting around {a['true_shooting_pct']}. "
        f"A Translation Score of {sc['translation']} places that adjusted profile above most D1 "
        f"{p['position']}s on the stats that matter most for the position.",
        "",
        "## Comparable Player Archetype",
        f"{archetype}.",
        "",
        "## Recommendation",
        f"**{rec}**: {reason}",
    ])


REPORT_SYSTEM = f"""You are a veteran college basketball scout writing for a D1 coaching staff. \
You write tight, specific, professional scouting reports.

Write the report in Markdown with exactly these level-2 headers, in this order:
{chr(10).join('## ' + s for s in SECTIONS)}

Rules:
- Cite ONLY numbers that appear in the JSON you are given, written the way they appear there \
(you may round to fewer decimals). Never compute new numbers, never estimate, never invent stats. \
If something is not in the data, describe it in words without a number.
- Snapshot: one line.
- Strengths and Weaknesses / Questions: 2-4 bullets each, grounded in the data. Weaknesses may \
include open questions a staff should answer on film.
- Projection at D1 Level: use the d1_adjusted numbers and percentiles.
- Comparable Player Archetype: describe a player TYPE. Never name a real player.
- Recommendation: start with **Pursue**, **Monitor**, or **Pass**, then one line of reasoning. \
suggested_recommendation is a rule of thumb; you may disagree if the data supports it.
- Under 350 words. No preamble before the first header."""


def llm_report(facts: dict[str, Any], feedback: str | None = None) -> str:
    """Ask Claude for the report. `feedback` is used for the one retry after a failed check."""
    user = "Player data (JSON):\n" + json.dumps(facts, indent=2)
    if feedback:
        user += f"\n\nYour previous draft had problems: {feedback}\nRewrite the full report fixing them."
    return complete(REPORT_SYSTEM, user, max_tokens=4000, effort="medium")


# --------------------------------------------------------------------------- cache
def _ensure_table(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS reports (
        player_id VARCHAR, input_hash VARCHAR, method VARCHAR, model VARCHAR,
        markdown VARCHAR, flagged VARCHAR, missing VARCHAR, created_at TIMESTAMP)""")


def get_cached(player_id: str, ihash: str, db_path: Path | str = DB_PATH) -> ReportResult | None:
    """Most recent cached report for this player+input, or None."""
    with get_connection(db_path) as con:
        _ensure_table(con)
        row = con.execute(
            "SELECT method, model, markdown, flagged, missing FROM reports "
            "WHERE player_id = ? AND input_hash = ? ORDER BY created_at DESC LIMIT 1",
            [player_id, ihash]).fetchone()
    if row is None:
        return None
    return ReportResult(player_id=player_id, method=row[0], model=row[1], markdown=row[2], cached=True,
                        flagged_numbers=json.loads(row[3]), missing_sections=json.loads(row[4]))


def save_cached(result: ReportResult, ihash: str, db_path: Path | str = DB_PATH) -> None:
    """Insert a report into the cache table."""
    with get_connection(db_path) as con:
        _ensure_table(con)
        con.execute("INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            result.player_id, ihash, result.method, result.model, result.markdown,
            json.dumps(result.flagged_numbers), json.dumps(result.missing_sections),
            datetime.now(timezone.utc).replace(tzinfo=None),
        ])


# --------------------------------------------------------------------------- entry point
def generate_report(row: pd.Series, force: bool = False, db_path: Path | str = DB_PATH,
                    use_llm: bool | None = None) -> ReportResult:
    """Return a cached report or generate a new one (LLM if available, else template)."""
    facts = build_report_input(row)
    ihash = input_hash(facts)
    use_llm = has_api_key() if use_llm is None else use_llm
    pid = str(row["player_id"])

    if not force:
        cached = get_cached(pid, ihash, db_path)
        # A cached offline template is upgraded once an API key is available.
        if cached and not (cached.method == "template" and use_llm):
            return cached

    result: ReportResult | None = None
    notes: list[str] = []
    if use_llm:
        try:
            md = llm_report(facts)
            flagged, missing = check_numbers(md, facts), missing_sections(md)
            if flagged or missing:
                problems = []
                if flagged:
                    problems.append(f"these numbers are not in the data: {', '.join(flagged)}")
                if missing:
                    problems.append(f"missing sections: {', '.join(missing)}")
                log.warning("Report check failed for %s (%s); retrying once", pid, "; ".join(problems))
                md = llm_report(facts, feedback="; ".join(problems))
                flagged, missing = check_numbers(md, facts), missing_sections(md)
                notes.append("First draft failed the fact check and was regenerated")
            result = ReportResult(player_id=pid, markdown=md, method="llm", model=SCOUT_MODEL,
                                  flagged_numbers=flagged, missing_sections=missing, notes=notes)
        except LLMError as e:
            notes.append(f"LLM unavailable ({e}); used the template report")
    else:
        notes.append("No API key set; used the template report")

    if result is None:
        md = template_report(facts, row)
        result = ReportResult(player_id=pid, markdown=md, method="template",
                              flagged_numbers=check_numbers(md, facts),
                              missing_sections=missing_sections(md), notes=notes)
    save_cached(result, ihash, db_path)
    return result
