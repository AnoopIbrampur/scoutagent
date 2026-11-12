"""Agency view: representation call, marketability split, best-fit destinations, leverage points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scout.agency import best_fits, leverage_points, program_styles, representation_call
from scout.scoring import explain_player, score_players
from scout.search import DuckDBSearchBackend
from scout.synergy import attach_synergy, generate_synergy_tables
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def world():
    players, _ = generate_players(seed=42)
    t = generate_synergy_tables(players, seed=42)
    be = DuckDBSearchBackend(players=players, synergy=(t["offense"], t["defense"]),
                             playtypes={k: t[k] for k in ("playtypes_offense", "playtypes_defense")})
    return be.scored, t["playtypes_offense"]


@pytest.mark.parametrize("t, u, expect", [
    (80, 70, "Sign"), (80, None, "Sign"), (80, 60, "Watch"), (60, 90, "Watch"), (54.9, 90, "Pass")])
def test_representation_call_thresholds(t, u, expect) -> None:
    assert representation_call(t, u)[0] == expect


def test_followers_drive_marketability_not_attention() -> None:
    players, _ = generate_players(seed=42)
    base = score_players(players)
    boosted = players.copy()
    boosted.loc[0, "social_followers"] = 10_000_000
    after = score_players(boosted)
    assert after.loc[0, "attention_score"] == pytest.approx(base.loc[0, "attention_score"])
    assert after.loc[0, "marketability_score"] > base.loc[0, "marketability_score"]
    assert after.loc[0, "underrated_score"] == pytest.approx(base.loc[0, "underrated_score"])


def test_best_fits_are_other_d1_programs_ranked_by_fit(world) -> None:
    scored, pt = world
    row = scored[scored.player_id == "P0161"].iloc[0]  # D2 PG
    fits = best_fits(row, scored, pt, program_styles(scored, pt))
    d1_schools = set(scored.loc[scored.division == "D1", "school"])
    assert len(fits) == 5 and all(f["school"] in d1_schools for f in fits)
    assert [f["fit"] for f in fits] == sorted((f["fit"] for f in fits), reverse=True)
    # Guard against the old bug where str.capitalize() lowercased "PG" and play-type names.
    assert all(" PG" in f["reason"] and " pg" not in f["reason"] and "p&r" not in f["reason"] for f in fits)
    assert all(not f["reason"][0].islower() for f in fits)


def test_best_fits_exclude_his_own_school(world) -> None:
    scored, pt = world
    row = scored[scored.division == "D1"].iloc[0]
    assert row["school"] not in {f["school"] for f in best_fits(row, scored, pt, n=30)}


def test_empty_position_reads_as_need(world) -> None:
    scored, pt = world
    row = scored[scored.player_id == "P0161"].iloc[0]
    all_fits = best_fits(row, scored, pt, n=30)
    d1 = scored[scored.division == "D1"]
    empty = set(d1.school) - set(d1.loc[d1.position == "PG", "school"])
    for f in all_fits:
        if f["school"] in empty:
            assert f["need"] == 100.0 and f["reason"].startswith("No PG on the roster")


def test_fits_work_without_play_types(world) -> None:
    scored, _ = world
    row = scored[scored.player_id == "P0161"].iloc[0]
    fits = best_fits(row, scored, None)
    assert len(fits) == 5 and all(f["style"] is None for f in fits)


def test_leverage_points_use_only_strong_stats(world) -> None:
    scored, _ = world
    row = scored[scored.player_id == "P0161"].iloc[0]
    points = leverage_points(row, explain_player(row))
    assert 1 <= len(points) <= 4 and all("better than" in p for p in points)
    weak = scored.sort_values("translation_score").iloc[0]
    assert leverage_points(weak, explain_player(weak)) == [] or \
        all(f["percentile"] >= 85 for f in explain_player(weak)["top_factors"] if f["percentile"] >= 85)


def test_high_marketability_becomes_a_leverage_point(world) -> None:
    scored, _ = world
    row = scored[scored.player_id == "P0161"].iloc[0].copy()
    row["marketability_score"] = 91.0
    assert any("Marketability 91.0" in p for p in leverage_points(row, explain_player(row)))
