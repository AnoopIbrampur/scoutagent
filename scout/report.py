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

PROMPT_VERSION = "v4"  # bump to invalidate cached reports after prompt changes
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
    unsupported_claims: list[str] = Field(default_factory=list)  # comparisons to players not in the data
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- input facts
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def suggested_recommendation(translation: float, underrated: float | None) -> str:
    """Rule of thumb shared by the template and the LLM prompt (unknown Underrated counts as met)."""
    if translation >= 75 and (underrated is None or underrated >= 65):
        return "Pursue"
    if translation >= 55:
        return "Monitor"
    return "Pass"


def build_report_input(row: pd.Series, playtypes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything the report may cite, already rounded the way it should be quoted.

    playtypes: his most-used offensive play types (from SearchBackend.playtype_summary).
    """
    exp = explain_player(row)
    weighted = [f["stat"] for f in exp["top_factors"]] + [exp["top_concern"]["stat"]]
    pct_stats = list(dict.fromkeys(weighted + ["ts_pct", "ast_to", "pts_per40", "reb_per40", "ast_per40", "tov_per40"]))
    t = round(float(row["translation_score"]), 1)
    u = None if pd.isna(row["underrated_score"]) else round(float(row["underrated_score"]), 1)
    opt_int = lambda v: None if pd.isna(v) else int(v)  # noqa: E731  (attention may be unknown)
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
        # 100 = best at the position for every stat, including turnovers and points allowed.
        "d1_position_percentiles": {STAT_LABELS[s]: int(round(float(row[f"pct_{s}"])))
                                    for s in pct_stats if not pd.isna(row[f"pct_{s}"])},
        "synergy": synergy_facts(row),
        "top_playtypes": playtypes or None,
        "attention": {
            "recruiting_stars": opt_int(row["recruiting_stars"]),
            "social_followers": opt_int(row["social_followers"]),
            "news_mentions_90d": opt_int(row["news_mentions_90d"]),
        },
        "scores": {"translation": t, "underrated": u,
                   "attention": None if pd.isna(row["attention_score"]) else round(float(row["attention_score"]), 1)},
        "explanation": {
            "top_factors": [f["text"] for f in exp["top_factors"]],
            "top_concern": exp["top_concern"]["text"],
            "notes": exp["notes"],
        },
        "suggested_recommendation": suggested_recommendation(t, u),
    }


def synergy_facts(row: pd.Series) -> dict[str, Any] | None:
    """Synergy All Possessions numbers, rounded as Synergy displays them (None if no data)."""
    if pd.isna(row.get("off_ppp", float("nan"))):
        return None
    out: dict[str, Any] = {
        "offense_possessions": int(row["syn_off_poss"]), "points_per_possession": round(float(row["off_ppp"]), 3),
        "ppp_rank": None if pd.isna(row["off_ppp_rank"]) else int(row["off_ppp_rank"]),
        "rating": row["off_rating"], "shot_quality_ssq": round(float(row["ssq"]), 2),
        "shot_making_ssm": round(float(row["ssm"]), 2), "turnover_rate": _pct(row["to_pct"]),
    }
    if not pd.isna(row.get("def_ppp", float("nan"))):
        out.update({
            "defense_possessions": int(row["syn_def_poss"]),
            "points_allowed_per_possession": round(float(row["def_ppp"]), 3),
            "defense_ppp_rank": None if pd.isna(row["def_ppp_rank"]) else int(row["def_ppp_rank"]),
            "defense_rating": row["def_rating"],
        })
    out["note"] = "Ranks are Synergy-style percentiles across all players, before any level adjustment."
    return out


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
    nums |= {abs(n) for n in nums}  # "-0.21 SSM" is matched as 0.21 by the number regex
    nums.update({40.0, 100.0})
    return nums


def _matches(token: str, allowed: set[float]) -> bool:
    """True if `token` equals some allowed number at the token's own precision."""
    x = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(round(a, decimals) == x for a in allowed)


_SMALL_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen"]
_TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_WORD_VALUE = {w: i for i, w in enumerate(_SMALL_WORDS)} | {w: 20 + 10 * i for i, w in enumerate(_TENS_WORDS)}
_WORD_NUM = re.compile(
    rf"\b(?:(?P<tens>{'|'.join(_TENS_WORDS)})(?:[- ](?P<unit>{'|'.join(_SMALL_WORDS[1:10])}))?"
    rf"|(?P<small>{'|'.join(_SMALL_WORDS)})|(?P<hundred>(?:a |one )?hundred))\b", re.IGNORECASE)
# Fractions and multiples are always computed quantities ("drops by a third",
# "twice as many"): the facts never contain them, so any use is flagged.
_FRACTION = re.compile(
    r"\b(?:(?:a|one|two)[- ](?:thirds?|quarters?|fifths?)|three[- ]quarters|half of|twice"
    r"|doubl(?:e|ed|es|ing)(?![- ](?:double|team|digit))|tripl(?:e|ed|es|ing)(?![- ]double))\b", re.IGNORECASE)
# Basketball idioms that contain number words but are not numeric claims.
# "one" is skipped entirely: "one of the best", "one season left" are too common.
_WORD_IDIOMS = [
    r"\b(?:two|three|four|five)[- ](?:way|guard|man|position|level)s?\b", r"\bstretch (?:four|five)\b",
    r"\bsmall[- ]ball (?:four|five)\b", r"\bstarting five\b", r"\bfrom three\b", r"\bone\b",
    r"\bthe (?:two|three|four|five)\b",  # positions: "play the four"
    r"\b(?:double|triple)[- ]doubles?\b",
]


def _word_number_tokens(text: str) -> list[tuple[str, float]]:
    """Spelled-out numbers in text as (original phrase, value), after removing idioms."""
    for pat in _WORD_IDIOMS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    out = []
    for m in _WORD_NUM.finditer(text):
        if m.group("hundred"):
            value = 100
        elif m.group("tens"):
            value = _WORD_VALUE[m.group("tens").lower()] + (_WORD_VALUE[m.group("unit").lower()] if m.group("unit") else 0)
        else:
            value = _WORD_VALUE[m.group("small").lower()]
        out.append((m.group(), float(value)))
    return out


def check_numbers(markdown: str, facts: dict[str, Any]) -> list[str]:
    """Return numbers in the report that are not supported by the input facts.

    Checks digits ("31.7"), heights ("6'4\""), spelled-out numbers ("eight" must
    round from a real value such as 7.6), and fractions/multiples ("a third",
    "twice"), which are always flagged because they are computed, not cited.
    """
    text = re.sub(r"\b(?:double|triple)[- ]doubles?\b", " ", markdown, flags=re.IGNORECASE)  # stat idioms
    flagged = [m.group() for m in _FRACTION.finditer(text)]
    text = _FRACTION.sub(" ", text)
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
    for phrase, value in _word_number_tokens(text):
        if not any(round(a) == value for a in allowed):
            flagged.append(phrase)
    return list(dict.fromkeys(flagged))


# Claims that rank the player against people outside his own data. Each report
# only sees one player, so "best on the board" or "top guard in the conference"
# cannot be supported. Comparisons to D1 percentiles are fine: those are in the input.
_SUPERLATIVE = r"(?:best|worst|highest|lowest|biggest|smallest|steepest|weakest|strongest|most|least|top|only)"
_OUTSIDE_SCOPE = (r"(?:(?:on|across|in|of) (?:this|the) (?:board|list|group|class|pool|shortlist)"
                  r"|among (?:this|the|these) (?:group|players|prospects|names|candidates)"
                  r"|in (?:the|his|this) (?:conference|league|division|country|nation|region|state)"
                  r"|(?:in|among) (?:the )?(?:nation|country)|nationally)")
_SENTENCE_CHAR = r"(?:[^.\n]|\.(?=\d))"  # anything but a sentence end; decimals like 65.7 stay intact
_COMPARISON = re.compile(
    rf"{_SENTENCE_CHAR}*\b{_SUPERLATIVE}\b{_SENTENCE_CHAR}*?\b{_OUTSIDE_SCOPE}\b{_SENTENCE_CHAR}*", re.IGNORECASE)


def check_comparisons(markdown: str) -> list[str]:
    """Return sentences ranking the player against others the input says nothing about."""
    return [m.group().replace("**", "").strip(" -*:") for m in _COMPARISON.finditer(markdown)]


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
_SKILL_OF_STAT = {"ast_to": "passing", "ast_per40": "passing", "three_pct_adj": "shooting", "ts_pct": "shooting",
                  "ssm": "shooting"}


def template_report(facts: dict[str, Any], row: pd.Series) -> str:
    """Deterministic offline report. Uses only values from `facts`, so it always passes the check."""
    p, s, d, a, sc, e = (facts[k] for k in ("player", "season_per_game", "derived", "d1_adjusted", "scores", "explanation"))
    att = facts["attention"]
    rec = facts["suggested_recommendation"]
    exp = explain_player(row)
    skill = _SKILL_OF_STAT.get(exp["top_factors"][0]["stat"], "other")
    archetype = _ARCHETYPES.get((p["position"], skill), _ARCHETYPES[(p["position"], "other")])
    reason = {
        "Pursue": (f"Translation {sc['translation']} with Underrated {sc['underrated']}: D1-level production the market has not priced in."
                   if sc["underrated"] is not None else
                   f"Translation {sc['translation']}: D1-level production; check how widely he is already known."),
        "Monitor": f"Translation {sc['translation']} is D1-viable but not a lock; see him against his best opponents.",
        "Pass": f"Translation {sc['translation']} does not project to a D1 rotation role yet.",
    }[rec]
    weaknesses = [f"- {e['top_concern']}"] + [f"- {n}" for n in e["notes"] if n.startswith("Small sample")]
    if att["recruiting_stars"] is None or att["news_mentions_90d"] is None:
        weaknesses.append("- Question: no attention data loaded, so how widely he is already known is "
                          "unclear. Verify the numbers in person.")
    else:
        stars = "no recruiting stars" if att["recruiting_stars"] == 0 else f"a {att['recruiting_stars']}-star rating"
        news = "no news mentions" if att["news_mentions_90d"] == 0 else f"{att['news_mentions_90d']} news mentions"
        weaknesses.append(f"- Question: {stars} and {news} in 90 days, so little outside film or "
                          f"evaluation exists. Verify the numbers in person.")
    syn = facts.get("synergy")
    synergy_line = ""
    if syn:
        synergy_line = (f" Synergy has him at {syn['points_per_possession']:.3f} points per possession "
                        f"({syn['rating'] or 'unrated'}) with shot-making of {syn['shot_making_ssm']:+.2f} over "
                        f"expected")
        if "points_allowed_per_possession" in syn:
            synergy_line += (f", and {syn['points_allowed_per_possession']:.3f} points allowed per possession "
                             f"on defense ({syn['defense_rating'] or 'unrated'})")
        synergy_line += "."
    if facts.get("top_playtypes"):
        parts = [f"{t['play_type']} ({t['share_of_possessions']} of possessions, {t['points_per_possession']:.3f} PPP"
                 + (f", {t['rating']})" if t["rating"] else ")") for t in facts["top_playtypes"]]
        synergy_line += " Most-used play types: " + "; ".join(parts) + "."
    return "\n".join([
        "## Snapshot",
        f"{p['class_year']} {p['height']} {p['position']} at {p['school']} ({p['division']}, {p['conference']}): "
        f"{s['points']} PPG, {s['rebounds']} RPG, {s['assists']} APG on {d['true_shooting_pct']} true shooting. "
        f"Translation {sc['translation']}"
        + (f", Underrated {sc['underrated']}." if sc["underrated"] is not None else " (Underrated not computed: no attention data)."),
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
        f"A Translation Score of {sc['translation']} places that adjusted profile "
        f"{'above' if sc['translation'] >= 50 else 'below'} most D1 "
        f"{p['position']}s on the stats that matter most for the position.{synergy_line}",
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
- If synergy data is present, use it: points per possession and its rating show efficiency per \
trip, SSM shows shot-making beyond the quality of his shots, and points allowed per possession \
shows how he defends. Percentiles in d1_position_percentiles are always "higher = better".
- If top_playtypes is present, describe his role from it (how he gets his offense and how \
efficient each is).
- Judge him only against the D1 percentiles provided. Do not compare him to other prospects, his conference, or any list or ranking you were not given.
- Write numbers as digits. Do not compute fractions or multiples ("a third", "twice").
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
    con.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS claims VARCHAR")  # added in v2


def get_cached(player_id: str, ihash: str, db_path: Path | str = DB_PATH) -> ReportResult | None:
    """Most recent cached report for this player+input, or None."""
    with get_connection(db_path) as con:
        _ensure_table(con)
        row = con.execute(
            "SELECT method, model, markdown, flagged, missing, claims FROM reports "
            "WHERE player_id = ? AND input_hash = ? ORDER BY created_at DESC LIMIT 1",
            [player_id, ihash]).fetchone()
    if row is None:
        return None
    return ReportResult(player_id=player_id, method=row[0], model=row[1], markdown=row[2], cached=True,
                        flagged_numbers=json.loads(row[3]), missing_sections=json.loads(row[4]),
                        unsupported_claims=json.loads(row[5] or "[]"))


def save_cached(result: ReportResult, ihash: str, db_path: Path | str = DB_PATH) -> None:
    """Insert a report into the cache table."""
    with get_connection(db_path) as con:
        _ensure_table(con)
        con.execute("INSERT INTO reports (player_id, input_hash, method, model, markdown, flagged, missing, "
                    "claims, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            result.player_id, ihash, result.method, result.model, result.markdown,
            json.dumps(result.flagged_numbers), json.dumps(result.missing_sections),
            json.dumps(result.unsupported_claims), datetime.now(timezone.utc).replace(tzinfo=None),
        ])


# --------------------------------------------------------------------------- entry point
def run_checks(markdown: str, facts: dict[str, Any]) -> dict[str, list[str]]:
    """All report validations, keyed by their ReportResult field names."""
    return {"flagged_numbers": check_numbers(markdown, facts),
            "missing_sections": missing_sections(markdown),
            "unsupported_claims": check_comparisons(markdown)}


def describe_problems(checks: dict[str, list[str]]) -> str:
    """Feedback for the retry prompt; empty string means the draft is clean."""
    problems = []
    if checks["flagged_numbers"]:
        problems.append(f"these numbers are not in the data: {', '.join(checks['flagged_numbers'])}")
    if checks["missing_sections"]:
        problems.append(f"missing sections: {', '.join(checks['missing_sections'])}")
    if checks["unsupported_claims"]:
        problems.append("these sentences compare him to players or rankings you were not given: "
                        + " | ".join(checks["unsupported_claims"]))
    return "; ".join(problems)


def generate_report(row: pd.Series, force: bool = False, db_path: Path | str = DB_PATH,
                    use_llm: bool | None = None, playtypes: list[dict[str, Any]] | None = None) -> ReportResult:
    """Return a cached report or generate a new one (LLM if available, else template)."""
    facts = build_report_input(row, playtypes)
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
            checks = run_checks(md, facts)
            if problems := describe_problems(checks):
                log.warning("Report check failed for %s (%s); retrying once", pid, problems)
                md = llm_report(facts, feedback=problems)
                checks = run_checks(md, facts)
                notes.append("First draft failed the fact check and was regenerated")
            result = ReportResult(player_id=pid, markdown=md, method="llm", model=SCOUT_MODEL,
                                  notes=notes, **checks)
        except LLMError as e:
            notes.append(f"LLM unavailable ({e}); used the template report")
    else:
        notes.append("No API key set; used the template report")

    if result is None:
        md = template_report(facts, row)
        result = ReportResult(player_id=pid, markdown=md, method="template", notes=notes,
                              **run_checks(md, facts))
    save_cached(result, ihash, db_path)
    return result
