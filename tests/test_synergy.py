"""Synergy-format tables: synthetic generator rules, PDF text parser, scoring hooks.

No real Synergy data here. The parser is tested by rendering a SYNTHETIC player
into the export's text layout and parsing it back.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from scout.scoring import score_players
from scout.synergy import (
    DEFENSE_SITUATIONS, OFFENSE_SITUATIONS, PERCENT_KEYS, RANK_KEYS, STAT_KEYS, attach_synergy,
    check_identities, generate_synergy, rating_for,
)
from scout.synergy_pdf import SynergyParseError, SynergyPdfSource, parse_row, parse_text, player_key
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def data():
    players, planted = generate_players(seed=42)
    offense, defense = generate_synergy(players, seed=42)
    return players, planted, offense, defense


# --- synthetic tables ------------------------------------------------------
def test_shape_and_situations(data) -> None:
    players, _, offense, defense = data
    assert len(offense) == len(defense) == 12 * len(players)
    assert list(offense[offense.player_id == "P0001"].situation) == OFFENSE_SITUATIONS
    assert list(defense[defense.player_id == "P0001"].situation) == DEFENSE_SITUATIONS
    assert list(offense.columns) == ["player_id", "situation"] + STAT_KEYS


def test_every_rule_holds(data) -> None:
    _, _, offense, defense = data
    assert check_identities(offense) == []
    assert check_identities(defense) == []


def test_offense_agrees_with_box_score(data) -> None:
    players, _, offense, _ = data
    allp = offense[offense.situation == "All Possessions"].set_index("player_id")
    season_pts = (players.set_index("player_id")["points_pg"] * players.set_index("player_id")["games"])
    # Rounding per-game inputs costs at most a few points over a season.
    assert ((allp["pts"] - season_pts).abs() <= players.set_index("player_id")["games"] * 0.15 + 3).all()


def test_ranks_ratings_and_small_samples(data) -> None:
    _, _, offense, defense = data
    ranked = offense.dropna(subset=["ppp_rank"])
    assert ranked["ppp_rank"].between(1, 100).all()
    assert (offense.loc[offense.poss < 10, "ppp_rank"].isna()).all()
    assert offense.loc[offense.situation == "All Possessions", "pct_time_rank"].isna().all()
    assert (ranked["rating"] == ranked["ppp_rank"].map(rating_for)).all()
    # Defense: allowing fewer points ranks higher.
    d = defense[(defense.situation == "All Possessions")].sort_values("ppp")
    assert d["ppp_rank"].iloc[0] > d["ppp_rank"].iloc[-1]


def test_rating_thresholds_match_real_examples() -> None:
    # Pairs seen on a real export: 91 Excellent, 73 Very Good, 27 Below Average, 1 Poor.
    assert [rating_for(r) for r in (91, 89, 73, 67, 27, 1)] == [
        "Excellent", "Excellent", "Very Good", "Very Good", "Below Average", "Poor"]


def test_deterministic(data) -> None:
    players, _, offense, _ = data
    pd.testing.assert_frame_equal(generate_synergy(players, seed=42)[0], offense)


def test_check_identities_catches_breaks(data) -> None:
    _, _, offense, _ = data
    broken = offense[offense.player_id == "P0001"].copy()
    broken.loc[broken.situation == "Transition", "poss"] += 3
    errors = check_identities(broken)
    assert any("Transition + Overall Half Court" in e for e in errors)
    assert any("PPP != PTS/POSS" in e for e in errors)


# --- PDF text parser (round trip through the export layout) -----------------
def _fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if key in PERCENT_KEYS:
        return f"{100 * v:.1f}%"
    if key in RANK_KEYS:
        return f"{v:.0f}%"
    if key == "ppp":
        return f"{v:.3f}"
    if key == "ssm":
        return f"{v:+.2f}"
    if key in ("ssq", "pps", "fta_per_fga"):
        return f"{v:.2f}"
    return str(v) if key == "rating" else f"{v:.0f}"


def render_export(table: pd.DataFrame, name: str, team: str, games: int) -> str:
    """A synthetic player's table in the exact text layout pdfplumber extracts from a real export."""
    lines = ["Player Details - Synergy Basketball 3/12/26, 12:21 AM", f"{name.upper()} - {team.upper()}, COLLEGE MEN",
             "Overall", "%TIME PPP PPP FG FG FG", "STAT POSS %TIME PTS PPP FG% EFG% SSQ PPS SSM ...", "RANK RANK RATING ATT MADE MISS"]
    for _, r in table.iterrows():
        icon = "" if r["situation"] in ("Overall Half Court", "Press") else ""
        lines.append(icon + r["situation"] + " " + " ".join(_fmt(k, r[k]) for k in STAT_KEYS))
    lines += [f"Games played:{games}", "https://apps.synergysports.com/basketball/players/... Page 1 of 1"]
    return "\n".join(lines)


@pytest.mark.parametrize("side", ["offense", "defense"])
def test_parser_round_trip(data, side) -> None:
    players, _, offense, defense = data
    table = (offense if side == "offense" else defense)
    table = table[table.player_id == "P0007"].reset_index(drop=True)
    exp = parse_text(render_export(table, "Jordan Testcase", "Example State", 29))
    assert (exp.player, exp.team, exp.side, exp.games) == ("Jordan Testcase", "Example State", side, 29)
    assert list(exp.table.situation) == list(table.situation)
    assert check_identities(exp.table.assign(player_id="x")) == []
    for key in ("poss", "pts", "fg_att", "fg_made", "tsa"):
        assert (exp.table[key] == table[key]).all(), key
    assert np.allclose(exp.table["ppp"], table["ppp"], equal_nan=True)
    assert list(exp.table["rating"].fillna("-")) == list(table["rating"].fillna("-"))


def test_parse_row_handles_two_word_rating_and_dashes() -> None:
    # Made-up numbers in the export's layout.
    row = parse_row("After Time Outs 60 15.0% 40% 66 1.100 70% Very Good 45 24 21 53.3% 55.6% 1.05 1.13 "
                    "+0.08 65.1% 51 8.3% 11.7% 0.29 10.0% 50.0%")
    assert row["rating"] == "Very Good" and row["ppp_rank"] == 70 and row["ssm"] == pytest.approx(0.08)
    assert row["to_pct"] == pytest.approx(0.083) and row["poss"] == 60
    empty = parse_row("Transition 0 0.0% - 0 - - - 0 0 0 - - - - - - 0 - - - - -")
    assert empty["poss"] == 0 and empty["rating"] is None and np.isnan(empty["ppp"])
    assert parse_row("Games played:30") is None


def test_parse_text_rejects_non_exports() -> None:
    with pytest.raises(SynergyParseError):
        parse_text("just some text\nnothing here")


def test_player_key() -> None:
    assert player_key("Jordan Testcase", "Example State") == "syn:jordan-testcase:example-state"


@pytest.mark.skipif(not os.getenv("SCOUT_SYNERGY_SAMPLE_DIR"),
                    reason="set SCOUT_SYNERGY_SAMPLE_DIR to a folder of real exports (kept outside the repo)")
def test_real_exports_parse_and_obey_rules() -> None:
    offense, defense, problems = SynergyPdfSource(os.environ["SCOUT_SYNERGY_SAMPLE_DIR"]).load()
    assert problems == []
    assert len(offense) % 12 == 0 and len(defense) % 12 == 0 and len(offense) > 0


# --- scoring hooks ---------------------------------------------------------
def test_attach_and_score(data) -> None:
    players, _, offense, defense = data
    scored = score_players(attach_synergy(players, offense, defense))
    assert scored[["pct_off_ppp", "pct_ssm", "pct_to_pct", "pct_def_ppp"]].notna().all().all()
    # Percentiles are "higher = better": the stingiest D1 defender ranks near 100.
    d1_pg = scored[(scored.division == "D1") & (scored.position == "PG")]
    assert d1_pg.loc[d1_pg["adj_def_ppp"].idxmin(), "pct_def_ppp"] > 90


def test_players_without_synergy_still_score(data) -> None:
    players, _, offense, defense = data
    partial = offense[offense.player_id != "P0001"], defense[defense.player_id != "P0001"]
    scored = score_players(attach_synergy(players, *partial)).set_index("player_id")
    assert np.isnan(scored.loc["P0001", "pct_off_ppp"])
    assert 0 <= scored.loc["P0001", "translation_score"] <= 100
    from scout.scoring import explain_player
    assert any("No Synergy data" in n for n in explain_player(scored.loc["P0001"])["notes"])
