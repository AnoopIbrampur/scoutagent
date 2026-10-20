"""Natural language -> structured search filters.

Strategy (most to least capable):
  1. LLM with a strict JSON schema (structured outputs), validated by Pydantic.
  2. If the output fails validation, retry once and tell the model what was wrong.
  3. If that fails too, or there is no API key, or the API errors: a
     regex/keyword parser that handles the common phrasings.

Every result records which path produced it, so the UI can show
"Here's how I understood your search" and whether the fallback was used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from scout.config import DIVISIONS, POSITIONS, has_api_key
from scout.llm import LLMError, complete

log = logging.getLogger(__name__)

Division = Literal["D1", "D2", "D3", "NAIA"]
Position = Literal["PG", "SG", "SF", "PF", "C"]
Emphasis = Literal["passing", "shooting", "scoring", "rebounding", "defense",
                   "rim_protection", "efficiency", "ball_security"]
SortBy = Literal["underrated", "translation"]
EMPHASIS_VALUES: tuple[str, ...] = Emphasis.__args__  # type: ignore[attr-defined]

LLMCall = Callable[[str, str], str]  # (system, user) -> raw JSON text


class SearchFilters(BaseModel):
    """Structured filters. Empty list / None means "no constraint"."""

    division: list[Division] = Field(default_factory=list)
    position: list[Position] = Field(default_factory=list)
    height_max_in: int | None = Field(default=None, ge=60, le=90)
    height_min_in: int | None = Field(default=None, ge=60, le=90)
    stat_emphasis: list[Emphasis] = Field(default_factory=list)
    sort_by: SortBy = "underrated"
    # Extra beyond the original spec: a school or player name to match with ILIKE.
    text_match: str | None = None

    @field_validator("division", "position", "stat_emphasis")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _height_order(self) -> "SearchFilters":
        if self.height_min_in and self.height_max_in and self.height_min_in > self.height_max_in:
            raise ValueError("height_min_in is greater than height_max_in")
        return self


class ParsedQuery(BaseModel):
    """Parser output plus provenance for the UI."""

    query: str
    filters: SearchFilters
    method: Literal["llm", "llm_retry", "keyword"]
    notes: list[str] = Field(default_factory=list)

    def chips(self) -> list[str]:
        """Short human-readable labels, one per active filter."""
        f = self.filters
        out = []
        if f.division:
            out.append("Division: " + ", ".join(f.division))
        if f.position:
            out.append("Position: " + ", ".join(f.position))
        if f.height_min_in:
            out.append(f"Height ≥ {inches_to_str(f.height_min_in)}")
        if f.height_max_in:
            out.append(f"Height ≤ {inches_to_str(f.height_max_in)}")
        if f.stat_emphasis:
            out.append("Emphasis: " + ", ".join(e.replace("_", " ") for e in f.stat_emphasis))
        if f.text_match:
            out.append(f'Matches "{f.text_match}"')
        out.append("Sorted by: " + ("Underrated" if f.sort_by == "underrated" else "Translation"))
        return out


def inches_to_str(inches: int) -> str:
    """74 -> 6'2\"."""
    return f"{inches // 12}'{inches % 12}\""


# --------------------------------------------------------------------------- LLM path
def _json_schema() -> dict:
    """JSON schema for structured outputs (mirrors SearchFilters)."""
    nullable_int = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "division": {"type": "array", "items": {"type": "string", "enum": list(DIVISIONS)}},
            "position": {"type": "array", "items": {"type": "string", "enum": list(POSITIONS)}},
            "height_max_in": nullable_int,
            "height_min_in": nullable_int,
            "stat_emphasis": {"type": "array", "items": {"type": "string", "enum": list(EMPHASIS_VALUES)}},
            "sort_by": {"type": "string", "enum": ["underrated", "translation"]},
            "text_match": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["division", "position", "height_max_in", "height_min_in",
                     "stat_emphasis", "sort_by", "text_match"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """You turn a college basketball recruiter's search request into JSON filters \
for a database of NCAA men's players.

Field rules:
- division: the level the player CURRENTLY plays at (D1, D2, D3, NAIA). Phrases like \
"translate to D1", "D1-ready", or "could play D1" describe the target level, not a filter. \
"Small school" or "lower division" means D2, D3 and NAIA. Leave empty if not stated.
- position: PG, SG, SF, PF, C. "Guard" = PG and SG. "Wing" = SG and SF. "Forward" = SF and PF. \
"Big" = PF and C. "Point guard"/"floor general" = PG. Leave empty if not stated.
- Heights are in inches (6'0" = 72). "Under 6'3"" means height_max_in 74. \
"6'3" or shorter" means 75. "At least 6'8"" means height_min_in 80. A bare height like \
"a 6'5 wing" means min 76, max 78.
- stat_emphasis: skills the user cares about, from the allowed list only. "Passer", \
"playmaker" = passing. "Shooter" = shooting. "Takes care of the ball", "low turnovers" = \
ball_security. "Rim protector", "shot blocker" = rim_protection. "Defender", "stopper" = defense.
- sort_by: "translation" only if the user wants the most D1-ready players regardless of hype. \
Otherwise "underrated" (the default: good players who are being overlooked).
- text_match: only if the user names a specific school or player, else null.

Never invent constraints the user did not ask for."""


def default_llm_call(system: str, user: str) -> str:
    """Production LLM call: structured JSON output, low effort (parsing is easy)."""
    return complete(system, user, max_tokens=2000, json_schema=_json_schema(), effort="low")


def _validate(raw: str) -> SearchFilters:
    return SearchFilters.model_validate(json.loads(raw))


# --------------------------------------------------------------------------- keyword path
_HEIGHT = r"(?P<ft>[5-7])\s*(?:'|’|ft\.?|feet|foot|-)\s*(?P<inch>1[01]|\d)?\s*(?:\"|”|''|in(?:ches)?\b)?"
_MAX_BEFORE = r"(?P<maxb>under|below|shorter than|less than|smaller than|at most|no taller than|up to|max(?:imum)?|<=?)"
_MIN_BEFORE = r"(?P<minb>over|above|taller than|at least|min(?:imum)?|>=?)"
_MAX_AFTER = r"\s*(?P<maxa>or (?:shorter|under|smaller|less|below)|and (?:under|below)|max)"
_MIN_AFTER = r"\s*(?P<mina>\+|or (?:taller|bigger|above|more)|and (?:up|above|taller))"

_DIVISION_RE = re.compile(r"\b(?:d|div(?:ision)?)[\s-]*(?P<n>iii|ii|i|1|2|3)\b|\b(?P<naia>naia)\b")
_TARGET_BEFORE = re.compile(r"(?:\bto|\bfor|\binto|\bplay|\bat the|\bmove up to|\bjump to)\s*(?:the\s+)?$")
_TARGET_AFTER = re.compile(r"^\s*-?\s*(?:ready|level|caliber|talent|potential|upside|translation|translat)")
_SMALL_SCHOOL_RE = re.compile(r"\b(?:small[- ]schools?|lower[- ]divisions?|non[- ]?d-?1|lower[- ]levels?|small[- ]colleges?)\b")

_POSITION_PATTERNS: list[tuple[str, list[str]]] = [
    (r"\bpoint guards?\b|\bpgs?\b|\blead guards?\b|\bfloor generals?\b", ["PG"]),
    (r"\bcombo guards?\b", ["PG", "SG"]),
    (r"\bshooting guards?\b|\bsgs?\b|\btwo[- ]guards?\b", ["SG"]),
    (r"\bsmall forwards?\b|\bsfs?\b", ["SF"]),
    (r"\bpower forwards?\b|\bpfs?\b|\bstretch (?:4|four)s?\b", ["PF"]),
    (r"\bcent(?:er|re)s?\b|\bbig m[ae]n\b|\bpost players?\b", ["C"]),
    (r"\bwings?\b", ["SG", "SF"]),
    (r"\bbigs?\b", ["PF", "C"]),
    (r"\bguards?\b", ["PG", "SG"]),
    (r"\bforwards?\b", ["SF", "PF"]),
]

_EMPHASIS_PATTERNS: list[tuple[str, str]] = [
    (r"pass(?:er|ing|es)?\b|playmak|assist|court vision|distribut|setting up", "passing"),
    (r"shoot|shooter|three|3[- ]?(?:pt|point)|sniper|marksman|spacing|jump ?shot", "shooting"),
    (r"scor(?:er|ing|e)\b|bucket|points\b", "scoring"),
    (r"rebound|glass|boards?\b", "rebounding"),
    (r"defen[cs]|steals?\b|lockdown|stopper|on-ball", "defense"),
    (r"rim protect|shot[- ]?block|blocks?\b|protects the rim", "rim_protection"),
    (r"efficien|true shooting|\bts%?", "efficiency"),
    (r"turnover|ball security|takes care of the ball|careful with the ball|low[- ]?tov|doesn'?t turn", "ball_security"),
]


def _parse_height(ft: str, inch: str | None) -> int:
    return int(ft) * 12 + int(inch or 0)


def _keyword_heights(q: str) -> tuple[int | None, int | None]:
    """Return (min_in, max_in) from phrases like "under 6'3", "6-8 or taller", "a 6'5 wing"."""
    hmin = hmax = None
    pattern = re.compile(rf"(?:{_MAX_BEFORE}|{_MIN_BEFORE})?\s*{_HEIGHT}(?:{_MAX_AFTER}|{_MIN_AFTER})?")
    for m in pattern.finditer(q):
        h = _parse_height(m.group("ft"), m.group("inch"))
        if m.group("maxb"):
            strict = m.group("maxb") in ("under", "below", "shorter than", "less than", "smaller than", "<")
            hmax = h - 1 if strict else h
        elif m.group("minb"):
            strict = m.group("minb") in ("over", "above", "taller than", ">")
            hmin = h + 1 if strict else h
        elif m.group("maxa"):
            hmax = h
        elif m.group("mina"):
            hmin = h
        else:  # bare height: allow +/- 1 inch
            hmin, hmax = h - 1, h + 1
    return hmin, hmax


def _keyword_divisions(q: str) -> list[str]:
    found: list[str] = []
    for m in _DIVISION_RE.finditer(q):
        if m.group("naia"):
            found.append("NAIA")
            continue
        if _TARGET_BEFORE.search(q[: m.start()]) or _TARGET_AFTER.search(q[m.end():]):
            continue  # "translate to D1", "D1-ready": target level, not a filter
        n = {"i": "1", "ii": "2", "iii": "3"}.get(m.group("n"), m.group("n"))
        found.append(f"D{n}")
    if _SMALL_SCHOOL_RE.search(q):
        found.extend(["D2", "D3", "NAIA"])
    return list(dict.fromkeys(found))


def _keyword_positions(q: str, original: str) -> list[str]:
    work, found = q, []
    for pat, positions in _POSITION_PATTERNS:
        if re.search(pat, work):
            found.extend(positions)
            work = re.sub(pat, " ", work)  # so "point guard" doesn't also match "guard"
    if re.search(r"(?<![\w'’])C(?![\w'’])", original):  # uppercase "C" as a position
        found.append("C")
    return list(dict.fromkeys(found))


def keyword_parse(query: str, known_schools: list[str] | None = None) -> SearchFilters:
    """Deterministic fallback parser. Handles common phrasings; never raises.

    known_schools: if given, a school name appearing in the query becomes text_match.
    """
    q = query.lower()
    school = next((s for s in sorted(known_schools or [], key=len, reverse=True) if s.lower() in q), None)
    hmin, hmax = _keyword_heights(q)
    if hmin and hmax and hmin > hmax:
        hmin = hmax = None
    sort_by = "translation" if re.search(
        r"\b(?:most|best|safest)\b.*\b(?:d-?1[- ]ready|ready|translat\w*)\b|\brank(?:ed)? by translation\b", q
    ) else "underrated"
    emphasis = [tag for pat, tag in _EMPHASIS_PATTERNS if re.search(pat, q)]
    clamp = lambda h: None if h is None else max(60, min(90, h))  # noqa: E731
    return SearchFilters(
        division=_keyword_divisions(q), position=_keyword_positions(q, query),
        height_min_in=clamp(hmin), height_max_in=clamp(hmax),
        stat_emphasis=emphasis, sort_by=sort_by, text_match=school,
    )


# --------------------------------------------------------------------------- entry point
def parse_query(query: str, llm: LLMCall | None = None, known_schools: list[str] | None = None) -> ParsedQuery:
    """Parse a search request. `llm` is injectable for tests; defaults to Claude if a key is set.

    known_schools only helps the keyword fallback (the LLM can extract names itself).
    """
    query = query.strip()
    notes: list[str] = []
    if llm is None:
        if has_api_key():
            llm = default_llm_call
        else:
            notes.append("No API key set, so the keyword parser was used")

    if llm is not None and query:
        user = f"Search request: {query}"
        try:
            result = ParsedQuery(query=query, filters=_validate(llm(SYSTEM_PROMPT, user)), method="llm")
            log.info("Parsed via LLM: %s", result.filters.model_dump())
            return result
        except (ValidationError, json.JSONDecodeError) as first_error:
            log.warning("LLM output failed validation, retrying once: %s", first_error)
            retry_user = (f"{user}\n\nYour previous answer was invalid: {str(first_error)[:500]}\n"
                          "Return corrected JSON only.")
            try:
                result = ParsedQuery(query=query, filters=_validate(llm(SYSTEM_PROMPT, retry_user)),
                                     method="llm_retry", notes=["First LLM answer was invalid; retried"])
                log.info("Parsed via LLM retry: %s", result.filters.model_dump())
                return result
            except (ValidationError, json.JSONDecodeError, LLMError) as e:
                notes.append(f"LLM output invalid twice ({type(e).__name__}); used keyword parser")
        except LLMError as e:
            notes.append(f"LLM unavailable ({e}); used keyword parser")

    result = ParsedQuery(query=query, filters=keyword_parse(query, known_schools), method="keyword", notes=notes)
    log.info("Parsed via keywords: %s", result.filters.model_dump())
    return result
