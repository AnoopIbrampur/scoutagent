"""Phase 1: synthetic data shape, realism, and the DuckDB round trip."""

from __future__ import annotations

import pandas as pd
import pytest

from scout.data_sources import ApifyDataSource, DataSourceError, SeedDataSource, validate_players
from scout.db import fetch_planted, fetch_players, get_connection, load_players
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def generated() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_players(seed=42)


def test_deterministic() -> None:
    a, pa = generate_players(seed=42)
    b, pb = generate_players(seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pa, pb)


def test_size_and_division_mix(generated) -> None:
    players, _ = generated
    assert len(players) == 600
    non_d1_share = (players["division"] != "D1").mean()
    assert 0.38 <= non_d1_share <= 0.42


def test_schema_contract(generated) -> None:
    players, _ = generated
    validate_players(players)  # raises on failure
    with pytest.raises(DataSourceError):
        validate_players(players.drop(columns=["fg_pct"]))


def test_planted_rows(generated) -> None:
    players, planted = generated
    gems = planted.merge(players, on="player_id").query("kind == 'gem'")
    assert len(gems) == 15
    assert (gems["division"] != "D1").all()
    assert gems["school"].is_unique
    assert gems["recruiting_stars"].max() <= 1
    # Headline demo query must have answers: D2 PG gems at 6'2" or shorter.
    assert len(gems.query("division == 'D2' and position == 'PG' and height_in <= 74")) >= 2


def test_position_realism(generated) -> None:
    players, _ = generated
    by_pos = players.groupby("position").mean(numeric_only=True)
    assert by_pos.loc["PG", "assists_pg"] > by_pos.loc["C", "assists_pg"] * 2
    assert by_pos.loc["C", "rebounds_pg"] > by_pos.loc["PG", "rebounds_pg"] * 1.5
    assert by_pos.loc["C", "blocks_pg"] > by_pos.loc["PG", "blocks_pg"] * 4
    assert by_pos.loc["C", "height_in"] > by_pos.loc["PG", "height_in"] + 6


def test_points_reconstruct_from_shots(generated) -> None:
    """Points were derived from shots, so the FGA approximation used in scoring should hold."""
    players, _ = generated
    ftm = players["ft_att_pg"] * players["ft_pct"]
    tpm = players["three_att_pg"] * players["three_pct"]
    fgm = (players["points_pg"] - ftm - tpm) / 2
    fga = fgm / players["fg_pct"]
    assert (fga > players["three_att_pg"] - 0.2).all()  # FGA must include the 3PA
    assert (fga < 30).all()


def test_attention_gap(generated) -> None:
    players, _ = generated
    med = players.groupby("division")["social_followers"].median()
    assert med["D1"] > 4 * med["D2"]


def test_duckdb_round_trip(generated, tmp_path) -> None:
    players, planted = generated
    csv = tmp_path / "players.csv"
    players.to_csv(csv, index=False)
    with get_connection(tmp_path / "t.duckdb") as con:
        assert load_players(con, SeedDataSource(csv).fetch_players(), planted) == 600
        assert len(fetch_planted(con)) == 25
        assert fetch_players(con)["player_id"].is_unique


def test_apify_stub_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ApifyDataSource().fetch_players()
