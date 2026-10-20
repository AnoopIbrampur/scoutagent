"""Transfer validation: can the harness recover known level factors, and does it reject bad files?"""

from __future__ import annotations

import pandas as pd
import pytest

from scout.config import SCORING_CONFIG
from scout.data_sources import DataSourceError
from scout.validation import (
    TRANSFER_COLUMNS, TRUE_FACTORS, calibrate_levels, load_transfers, no_level_adjustment, predictive_check,
    rotation_hit_rates, score_transfers, simulate_transfers, spearman,
)
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def sim():
    players, _ = generate_players(seed=42)
    t = simulate_transfers(players)
    return players, t, score_transfers(t, players)


def test_recovers_planted_level_factors(sim) -> None:
    _, t, _ = sim
    for c in calibrate_levels(t):
        truth = TRUE_FACTORS[c.division]
        assert abs(c.suggested - truth) < 0.03, (c.division, c.suggested, truth)
        assert c.suggested_ci[0] <= truth <= c.suggested_ci[1], (c.division, c.suggested_ci, truth)
        assert c.configured != truth  # the test is only meaningful if config and truth differ


def test_translation_predicts_d1_outcomes(sim) -> None:
    _, _, scored = sim
    res = {(r.predictor, r.outcome): r for r in predictive_check(scored)}
    assert res["translation", "d1_minutes_pg"].ci[0] > 0
    assert res["translation", "d1_production"].rho > res["raw_ppg", "d1_production"].rho
    hits = rotation_hit_rates(scored)
    assert hits.loc["Top third", "rotation_rate"] > hits.loc["Bottom third", "rotation_rate"]


def test_level_adjustment_switch_restores_config() -> None:
    before = dict(SCORING_CONFIG["level_factor"])
    with no_level_adjustment():
        assert set(SCORING_CONFIG["level_factor"].values()) == {1.0}
    assert SCORING_CONFIG["level_factor"] == before


def test_spearman_matches_definition() -> None:
    assert spearman(pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 40])) == pytest.approx(1.0)
    assert spearman(pd.Series([1, 2, 3, 4]), pd.Series([4, 3, 2, 1])) == pytest.approx(-1.0)


def test_example_file_loads() -> None:
    t = load_transfers("docs/transfers_example.csv")
    assert list(t.columns) == TRANSFER_COLUMNS and t.loc[1, "post_games"] == 0


def test_rejects_d1_origin_and_blank_cells(tmp_path) -> None:
    t = pd.read_csv("docs/transfers_example.csv")
    t.loc[0, "from_division"] = "D1"
    t.loc[0, "pre_points_pg"] = None
    t.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(DataSourceError, match="from_division") as e:
        load_transfers(tmp_path / "bad.csv")
    assert "pre_points_pg" in str(e.value)


def test_player_who_never_played_counts_as_zero_minutes(sim) -> None:
    players, _, _ = sim
    t = load_transfers("docs/transfers_example.csv")
    scored = score_transfers(t, players)
    assert scored.loc[1, "d1_minutes_pg"] == 0 and pd.isna(scored.loc[1, "d1_production"])
