"""Real-player path: roster CSV, template from exports, baseline scoring, real dataset mode.

Uses synthetic players dressed up as "real" ones. No real data in the repo.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scout.data_sources import DataSourceError
from scout.db import get_connection, load_players, load_real_players, load_synergy
from scout.roster import ROSTER_COLUMNS, RosterCsvSource, parse_height, roster_template
from scout.scoring import explain_player, score_players
from scout.search import DuckDBSearchBackend
from scout.synergy import attach_synergy, generate_synergy_tables
from scout.synergy_pdf import parse_text, player_key
from scripts.generate_seed_data import generate_players
from tests.test_synergy import render_export


@pytest.fixture(scope="module")
def synth():
    players, planted = generate_players(seed=42)
    return players, planted, generate_synergy_tables(players, seed=42)


def _as_real(players: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Five non-D1 synthetic players as a roster CSV would describe them (attention blank)."""
    df = players[players.division != "D1"].head(n).drop(columns="player_id").copy()
    df[["recruiting_stars", "social_followers", "news_mentions_90d"]] = np.nan
    return df[ROSTER_COLUMNS]


@pytest.mark.parametrize("text, inches", [("6'2\"", 74), ("6-2", 74), ("6 2", 74), ("74", 74), ("7'0", 84), ("6’10”", 82)])
def test_parse_height(text, inches) -> None:
    assert parse_height(text) == inches


def test_parse_height_rejects_garbage() -> None:
    with pytest.raises(DataSourceError):
        parse_height("tall")


def test_roster_csv_round_trip(synth, tmp_path) -> None:
    real = _as_real(synth[0])
    real["height_in"] = real["height_in"].map(lambda h: f"{h // 12}'{h % 12}\"")
    path = tmp_path / "roster.csv"
    real.drop(columns=["conference_strength"]).to_csv(path, index=False)
    out = RosterCsvSource(path).fetch_players()
    assert list(out["player_id"]) == [player_key(n, s) for n, s in zip(real["name"], real["school"])]
    assert (out["conference_strength"] == 0.5).all()  # default when the column is missing
    assert out["recruiting_stars"].isna().all()  # blank attention = unknown


def test_roster_lists_blank_cells(synth, tmp_path) -> None:
    real = _as_real(synth[0], 2)
    real.loc[real.index[0], "minutes_pg"] = np.nan
    real.to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(DataSourceError, match="minutes_pg"):
        RosterCsvSource(tmp_path / "r.csv").fetch_players()


def test_template_prefill_matches_box_score(synth) -> None:
    players, _, t = synth
    p = players.iloc[3]
    off = t["offense"][t["offense"].player_id == p.player_id].reset_index(drop=True)
    exp = parse_text(render_export(off, "Jordan Testcase", "Example State", int(p.games)))
    row = roster_template([exp]).iloc[0]
    assert row["name"] == "Jordan Testcase" and row["games"] == p.games
    assert abs(row["points_pg"] - p.points_pg) <= 0.2
    assert abs(row["fg_pct"] - p.fg_pct) <= 0.006
    assert abs(row["turnovers_pg"] - p.turnovers_pg) <= 0.2
    assert row["height_in"] == ""  # left for the user


def test_reference_baseline_and_unknown_attention(synth) -> None:
    players, _, t = synth
    real = _as_real(players).assign(player_id=lambda d: [f"syn:real-{i}" for i in range(len(d))])
    ref = attach_synergy(players[players.division == "D1"], t["offense"], t["defense"])
    scored = score_players(real, reference=ref, baseline_label="synthetic D1")
    assert len(scored) == len(real) and set(scored.player_id) == set(real.player_id)
    assert scored["translation_score"].between(0, 100).all()
    assert scored["underrated_score"].isna().all()
    notes = explain_player(scored.iloc[0])["notes"]
    assert any("synthetic D1 baseline" in n for n in notes) and any("Underrated not computed" in n for n in notes)


def test_real_dataset_mode_end_to_end(synth, tmp_path) -> None:
    players, planted, t = synth
    db = tmp_path / "real.duckdb"
    real = _as_real(players)
    real["player_id"] = [player_key(n, s) for n, s in zip(real["name"], real["school"])]
    renamed = {pid: key for pid, key in zip(players[players.division != "D1"].head(5).player_id, real.player_id)}
    with get_connection(db) as con:
        load_players(con, players, planted)
        load_synergy(con, t["offense"], t["defense"],
                     playtypes={k: t[k] for k in ("playtypes_offense", "playtypes_defense")})
        load_real_players(con, real)
        real_tables = {k: v[v.player_id.isin(renamed)].assign(player_id=lambda d: d.player_id.map(renamed))
                       for k, v in t.items()}
        load_synergy(con, real_tables["offense"], real_tables["defense"], prefix="synergy_real",
                     playtypes={k: real_tables[k] for k in ("playtypes_offense", "playtypes_defense")})
    be = DuckDBSearchBackend(db, dataset="real")
    assert be.dataset == "real" and len(be.scored) == 5
    res = be.search(be_filters(), limit=10, apply_gate=False)
    json.dumps(res, allow_nan=False)  # unknown Underrated must serialize as null, not NaN
    pid = res["results"][0]["player_id"]
    assert be.get_synergy(pid)["playtypes_offense"] and be.playtype_summary(pid)


def be_filters():
    from scout.query_parser import SearchFilters
    return SearchFilters()
