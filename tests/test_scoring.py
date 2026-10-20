"""Phase 2: scoring math, adjustments, explanations, and the planted-gem acceptance test."""

from __future__ import annotations

import pandas as pd
import pytest

from scout.config import POSITION_WEIGHTS
from scout.scoring import (
    adjust_stats, derive_stats, estimate_fga, explain_player, level_multiplier,
    percentile_against, score_players, shrink_three_pct, true_shooting, underrated_score,
)
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def seed() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_players(seed=42)


@pytest.fixture(scope="module")
def scored(seed) -> pd.DataFrame:
    players, _ = seed
    return score_players(players)


def _row(**overrides) -> pd.DataFrame:
    base = dict(points_pg=13.2, fg_pct=0.44, three_att_pg=4.0, three_pct=0.35,
                ft_att_pg=4.0, ft_pct=0.75)
    base.update(overrides)
    return pd.DataFrame([base])


# --- derived stats ---------------------------------------------------------
def test_estimate_fga_inverts_points_identity() -> None:
    # 10 FGA: 6 twos at 50% (3 made), 4 threes at 35% (1.4 made); 4 FTA at 75% (3 made)
    # PTS = 2*3 + 3*1.4 + 3 = 13.2 ; FG% = 4.4/10
    assert estimate_fga(_row()).iloc[0] == pytest.approx(10.0)


def test_estimate_fga_never_below_three_attempts() -> None:
    assert estimate_fga(_row(points_pg=0.0, fg_pct=0.0)).iloc[0] == pytest.approx(4.0)


def test_true_shooting_formula() -> None:
    ts = true_shooting(pd.Series([20.0]), pd.Series([15.0]), pd.Series([5.0]))
    assert ts.iloc[0] == pytest.approx(20 / (2 * (15 + 0.44 * 5)))


def test_true_shooting_zero_attempts_is_zero() -> None:
    assert true_shooting(pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0])).iloc[0] == 0.0


def test_per40_and_ast_to(seed) -> None:
    players, _ = seed
    df = derive_stats(players)
    r = df.iloc[0]
    assert r["pts_per40"] == pytest.approx(r["points_pg"] * 40 / r["minutes_pg"])
    assert r["ast_to"] == pytest.approx(r["assists_pg"] / max(r["turnovers_pg"], 0.1))


def test_three_pct_shrinkage_depends_on_volume(seed) -> None:
    players, _ = seed
    df = players.copy()
    df.loc[0, ["three_att_pg", "three_pct", "games", "division"]] = [0.2, 0.6, 30, "D2"]  # 6 attempts
    df.loc[1, ["three_att_pg", "three_pct", "games", "division"]] = [8.0, 0.6, 30, "D2"]  # 240 attempts
    adj = shrink_three_pct(df)
    assert adj.iloc[0] < 0.42 < adj.iloc[1]  # low volume pulled to ~average, high volume stays high


# --- adjustments -----------------------------------------------------------
def test_level_multiplier_anchors() -> None:
    assert level_multiplier("D1", 0.5) == pytest.approx(1.0)
    assert level_multiplier("D2", 0.5) == pytest.approx(0.8)
    assert level_multiplier("D3", 1.0) == pytest.approx(0.65 * 1.15)


def test_efficiency_adjustment_only_discounts_strengths(seed) -> None:
    players, _ = seed
    df = adjust_stats(derive_stats(players))
    d1_avg = df[(df.division == "D1") & (df.position == "PG")]["ts_pct"].mean()
    low_d2 = df[(df.division == "D2") & (df.ts_pct < d1_avg) & (df.position == "PG")]
    high_d2 = df[(df.division == "D2") & (df.ts_pct > d1_avg) & (df.position == "PG")]
    assert (low_d2["adj_ts_pct"] == low_d2["ts_pct"]).all()
    assert (high_d2["adj_ts_pct"] < high_d2["ts_pct"]).all()
    strong_d1 = df[(df.division == "D1") & (df.level_mult > 1)]
    assert (strong_d1["adj_ts_pct"] == strong_d1["ts_pct"]).all()  # trust capped at 1


def test_turnovers_inflated_for_lower_levels(seed) -> None:
    players, _ = seed
    df = adjust_stats(derive_stats(players))
    d2 = df[df.division == "D2"]
    assert (d2["adj_tov_per40"] > d2["tov_per40"]).all()


# --- percentiles and scores ------------------------------------------------
def test_percentile_against_with_ties() -> None:
    ref = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    out = percentile_against(pd.Series([2.5, 5.0, 2.0, 0.0]), ref)
    assert list(out) == [50.0, 100.0, 37.5, 0.0]


def test_position_weights_sum_to_one() -> None:
    for pos, weights in POSITION_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), pos


def test_underrated_rescaling() -> None:
    out = underrated_score(pd.Series([50.0, 100.0, 0.0]), pd.Series([50.0, 0.0, 100.0]))
    assert list(out) == [50.0, 100.0, 0.0]


def test_scores_bounded(scored) -> None:
    for col in ("translation_score", "attention_score", "underrated_score"):
        assert scored[col].between(0, 100).all(), col


def test_same_line_scores_lower_at_lower_level(seed) -> None:
    players, _ = seed
    pid = players[(players.division == "D1") & (players.position == "PG")].iloc[0]["player_id"]
    df = players.copy()
    clone = df[df.player_id == pid].copy()
    clone["player_id"], clone["division"] = "CLONE", "D2"
    out = score_players(pd.concat([df, clone], ignore_index=True)).set_index("player_id")
    assert out.loc["CLONE", "translation_score"] < out.loc[pid, "translation_score"]


def test_better_efficiency_raises_translation(seed) -> None:
    players, _ = seed
    i = players.index[(players.division == "D2") & (players.position == "SG")][0]
    worse, better = players.copy(), players.copy()
    better.loc[i, "points_pg"] += 3.0  # same attempts, more points -> higher TS%
    before = score_players(worse).loc[i, "translation_score"]
    after = score_players(better).loc[i, "translation_score"]
    assert after > before


def test_more_attention_raises_attention_score(seed) -> None:
    players, _ = seed
    df = players.copy()
    i = df.index[0]
    base = score_players(df).loc[i, "attention_score"]
    df.loc[i, ["recruiting_stars", "social_followers", "news_mentions_90d"]] = [5, 10**6, 100]
    assert score_players(df).loc[i, "attention_score"] > base


# --- explanations ----------------------------------------------------------
def test_explanation_structure(scored) -> None:
    exp = explain_player(scored.iloc[0])
    assert len(exp["top_factors"]) == 3
    contribs = [f["contribution"] for f in exp["top_factors"]]
    assert contribs == sorted(contribs, reverse=True)
    assert exp["top_concern"]["text"]
    assert any("Level multiplier" in n for n in exp["notes"])


def test_turnover_concern_wording(scored) -> None:
    sloppy = scored.sort_values("pct_tov_per40", ascending=False).iloc[0]
    exp = explain_player(sloppy)
    assert exp["top_concern"]["stat"] == "tov_per40"
    assert "higher than" in exp["top_concern"]["text"]


# --- acceptance ------------------------------------------------------------
def test_acceptance_planted_gems_rank_in_top_50(seed, scored) -> None:
    _, planted = seed
    top50 = set(scored.nlargest(50, "underrated_score")["player_id"])
    gems = set(planted.loc[planted.kind == "gem", "player_id"])
    assert len(gems & top50) >= 10
