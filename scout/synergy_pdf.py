"""Read Synergy "Player Details > Overall" PDF exports into Synergy-format tables.

Workflow: someone with a Synergy account prints/exports a player's Overall page
(offense and defense) to PDF and drops the files in a folder. This module turns
each file into rows matching scout.synergy.SYNERGY_COLUMNS.

Real exports are licensed data: keep them OUT of git (data/ is gitignored).
Tests use made-up numbers in the same text layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scout.synergy import (
    ALL_PLAYTYPES, ALL_SITUATIONS, OFFENSE_SITUATIONS, PERCENT_KEYS, PLAYTYPE_ALIASES, STAT_KEYS,
    check_identities, check_playtypes,
)

_RATING_WORDS = {"Excellent", "Good", "Average", "Poor"}
_RATING_PREFIX = {"Very", "Below", "Above"}
_HEADER = re.compile(r"^(?P<name>.+?) - (?P<team>.+?), (?P<comp>COLLEGE (?:MEN|WOMEN).*)$")
_GAMES = re.compile(r"Games played:\s*(\d+)")
_ICON = re.compile(r"[\ue000-\uf8ff]")
# Row labels (Overall view + play types + play-type aliases), longest first so
# "Overall Half Court" doesn't match a shorter label. Aliases map to canonical names.
_CANONICAL = {n: n for n in ALL_SITUATIONS | ALL_PLAYTYPES} | PLAYTYPE_ALIASES
_SITUATIONS = sorted(_CANONICAL, key=len, reverse=True)
_OVERALL_ONLY = set(ALL_SITUATIONS) - {"Transition"}


TABLE_KEYS = ("offense", "defense", "playtypes_offense", "playtypes_defense")


class SynergyParseError(ValueError):
    """The text doesn't look like a Synergy Overall export."""


@dataclass
class SynergyExport:
    """One parsed PDF: a player's offense OR defense table."""

    player: str
    team: str
    side: str  # "offense" or "defense"
    view: str  # "overall" or "playtypes"
    games: int | None
    table: pd.DataFrame  # columns: situation + STAT_KEYS


def _value(token: str, key: str) -> float | str | None:
    """Convert one cell: '61.5%' -> 0.615 (or 61 for ranks), '+0.07' -> 0.07, '-' -> NaN."""
    if token == "-":
        return float("nan")
    if key == "rating":
        return token
    num = float(token.rstrip("%").replace(",", ""))
    if key in PERCENT_KEYS:
        return num / 100
    return num  # ranks stay 0-100, counts and rates as-is


def parse_row(line: str) -> dict | None:
    """Parse one table line, or return None if it isn't a situation row."""
    situation = next((s for s in _SITUATIONS if line.startswith(s + " ")), None)
    if situation is None:
        return None
    tokens = line[len(situation):].split()
    head, rest = tokens[:6], tokens[6:]
    if rest and rest[0] in _RATING_PREFIX:
        rating, rest = f"{rest[0]} {rest[1]}", rest[2:]
    elif rest and (rest[0] in _RATING_WORDS or rest[0] == "-"):
        rating, rest = rest[0], rest[1:]
    else:
        raise SynergyParseError(f"no RATING in row: {line!r}")
    cells = head + [rating] + rest
    if len(cells) != len(STAT_KEYS):
        raise SynergyParseError(f"expected {len(STAT_KEYS)} cells, got {len(cells)}: {line!r}")
    row = {"situation": _CANONICAL[situation]}
    for key, token in zip(STAT_KEYS, cells):
        row[key] = _value(token, key)
    row["rating"] = None if row["rating"] == "-" or pd.isna(row["rating"]) else row["rating"]
    return row


def parse_text(text: str) -> SynergyExport:
    """Parse the extracted text of one export."""
    # Rows with sub-rows start with an expand-arrow icon from a private-use font (e.g. U+F0DD).
    lines = [_ICON.sub("", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    header = next((m for m in map(_HEADER.match, lines) if m), None)
    if header is None:
        raise SynergyParseError("no 'PLAYER - TEAM, COLLEGE MEN' header line")
    rows = [r for r in map(parse_row, lines) if r is not None]
    if not rows:
        raise SynergyParseError("no situation rows found")
    table = pd.DataFrame(rows)
    for key in STAT_KEYS:
        if key != "rating":
            table[key] = pd.to_numeric(table[key])
    names = set(table["situation"])
    view = "overall" if names & _OVERALL_ONLY else "playtypes"
    # Defense: the Overall view has "Man Defense"; the export URL ends in offense=false.
    side = "defense" if "Man Defense" in names or "offense=false" in text else "offense"
    games = _GAMES.search(text)
    return SynergyExport(player=header["name"].title(), team=header["team"].title(), side=side, view=view,
                         games=int(games.group(1)) if games else None, table=table)


def parse_pdf(path: Path | str) -> SynergyExport:
    """Extract text from a PDF (all pages) and parse it."""
    import pdfplumber  # imported lazily: only needed when reading real exports

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return parse_text(text)


def player_key(player: str, team: str) -> str:
    """Stable ID for a real player parsed from exports, e.g. 'syn:jane-doe:state-u'."""
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")  # noqa: E731
    return f"syn:{slug(player)}:{slug(team)}"


class SynergyPdfSource:
    """All Synergy PDFs in a folder -> (offense, defense) tables keyed by player_key.

    This supplies Synergy tables only. A full real-data pipeline still needs a
    box-score/roster source (height, position, division, per-game stats) behind
    the DataSource interface; rows are joined on player_id.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Overall view only: (offense, defense, problems). See load_all for play types too."""
        tables, problems, _ = self.load_all()
        return tables["offense"], tables["defense"], problems

    def load_all(self) -> tuple[dict[str, pd.DataFrame], list[str], list[SynergyExport]]:
        """All exports in the folder.

        Returns (tables, problems, exports). tables has keys offense, defense,
        playtypes_offense, playtypes_defense (empty if none). Problems include
        parse failures, row-rule violations, and play types that don't add up to
        the Overall table.
        """
        cols = ["player_id", "situation"] + STAT_KEYS
        frames: dict[str, list[pd.DataFrame]] = {k: [] for k in TABLE_KEYS}
        problems: list[str] = []
        exports: list[SynergyExport] = []
        for path in sorted(self.folder.glob("*.pdf")):
            try:
                exp = parse_pdf(path)
            except SynergyParseError as e:
                problems.append(f"{path.name}: {e}")
                continue
            exports.append(exp)
            key = exp.side if exp.view == "overall" else f"playtypes_{exp.side}"
            frames[key].append(exp.table.assign(player_id=player_key(exp.player, exp.team))[cols])
        tables = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=cols) for k, v in frames.items()}
        for side in ("offense", "defense"):
            if not tables[side].empty:
                problems += [f"{side}: {e}" for e in check_identities(tables[side])]
            pt = tables[f"playtypes_{side}"]
            if not pt.empty:
                problems += [f"playtypes_{side}: {e}" for e in check_playtypes(pt, tables[side])]
        return tables, problems, exports
