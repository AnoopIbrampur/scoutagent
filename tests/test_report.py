"""Phase 4: report fact check, template fallback, LLM retry, and caching."""

from __future__ import annotations

import pandas as pd
import pytest

import scout.report as report
from scout.llm import LLMError
from scout.report import (
    SECTIONS, build_report_input, check_comparisons, check_numbers, generate_report, missing_sections,
    template_report,
)
from scout.search import DuckDBSearchBackend
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    players, _ = generate_players(seed=42)
    return DuckDBSearchBackend(players=players).scored


@pytest.fixture(scope="module")
def row(scored) -> pd.Series:
    return scored.sort_values("underrated_score", ascending=False).iloc[0]


@pytest.fixture
def facts(row) -> dict:
    return build_report_input(row)


# --- number check ----------------------------------------------------------
def test_template_passes_check_for_every_player(scored) -> None:
    for _, r in scored.iterrows():
        f = build_report_input(r)
        md = template_report(f, r)
        assert check_numbers(md, f) == [], r["player_id"]
        assert missing_sections(md) == []


def test_invented_numbers_are_flagged(facts) -> None:
    md = "## Snapshot\nHe averaged 31.7 PPG and had 14 double-doubles."
    assert check_numbers(md, facts) == ["31.7", "14"]


def test_rounding_is_allowed(facts) -> None:
    ts = facts["derived"]["true_shooting_pct"]  # e.g. "63.6%"
    rounded = str(round(float(ts.rstrip("%"))))
    assert check_numbers(f"True shooting of {ts}, call it {rounded}%.", facts) == []


def test_wrong_height_is_flagged(facts) -> None:
    good = facts["player"]["height"]
    assert check_numbers(f"A {good} guard.", facts) == []
    assert check_numbers("A 6'9\" guard.", facts) == ["6'9\""]


def test_structural_numbers_not_flagged(facts) -> None:
    md = "Projects at D1 per 40 minutes; a 3-and-D wing with three-point range and 3P% upside."
    assert check_numbers(md, facts) == []


def test_percentile_complement_allowed(facts) -> None:
    pct = next(iter(facts["d1_position_percentiles"].values()))
    assert check_numbers(f"{pct}th percentile, top {100 - pct}% at the position", facts) == []


def test_missing_sections_detected() -> None:
    assert missing_sections("## Snapshot\nx\n## Strengths\ny") == SECTIONS[2:]


# --- generation paths ------------------------------------------------------
def _good_llm_md(facts: dict) -> str:
    sc = facts["scores"]
    return "\n\n".join(f"## {s}\nTranslation {sc['translation']}." for s in SECTIONS[:-1]) + \
        "\n\n## Recommendation\n**Pursue**: strong."


def test_no_key_uses_template_and_caches(row, tmp_path) -> None:
    db = tmp_path / "r.duckdb"
    first = generate_report(row, db_path=db, use_llm=False)
    assert first.method == "template" and not first.cached
    second = generate_report(row, db_path=db, use_llm=False)
    assert second.cached and second.markdown == first.markdown


def test_llm_success(row, tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(report, "llm_report", lambda f, feedback=None: calls.append(feedback) or _good_llm_md(f))
    out = generate_report(row, db_path=tmp_path / "r.duckdb", use_llm=True)
    assert out.method == "llm" and out.flagged_numbers == [] and calls == [None]


def test_llm_flagged_draft_is_retried(row, tmp_path, monkeypatch) -> None:
    drafts = iter(["## Snapshot\nScored 99.9 PPG.", None])
    calls = []

    def fake(f, feedback=None):
        calls.append(feedback)
        d = next(drafts)
        return d if d else _good_llm_md(f)

    monkeypatch.setattr(report, "llm_report", fake)
    out = generate_report(row, db_path=tmp_path / "r.duckdb", use_llm=True)
    assert len(calls) == 2 and "99.9" in calls[1]
    assert out.flagged_numbers == [] and out.missing_sections == []
    assert any("regenerated" in n for n in out.notes)


def test_llm_error_falls_back_to_template(row, tmp_path, monkeypatch) -> None:
    def boom(f, feedback=None):
        raise LLMError("Rate limited")

    monkeypatch.setattr(report, "llm_report", boom)
    out = generate_report(row, db_path=tmp_path / "r.duckdb", use_llm=True)
    assert out.method == "template" and any("Rate limited" in n for n in out.notes)


def test_cached_template_upgraded_when_llm_available(row, tmp_path, monkeypatch) -> None:
    db = tmp_path / "r.duckdb"
    generate_report(row, db_path=db, use_llm=False)
    monkeypatch.setattr(report, "llm_report", lambda f, feedback=None: _good_llm_md(f))
    assert generate_report(row, db_path=db, use_llm=True).method == "llm"
    assert generate_report(row, db_path=db, use_llm=True).cached  # LLM report now cached


def test_changed_input_invalidates_cache(row, tmp_path) -> None:
    db = tmp_path / "r.duckdb"
    generate_report(row, db_path=db, use_llm=False)
    changed = row.copy()
    changed["points_pg"] = row["points_pg"] + 1
    assert not generate_report(changed, db_path=db, use_llm=False).cached


# --- spelled-out numbers and unsupported comparisons ----------------------
# Fixtures are real slips from a test round where Claude wrote reports for the
# top 5 players. Each passed the digit-only check but should not have.
@pytest.fixture(scope="module")
def player_facts(scored):
    return lambda pid: build_report_input(scored[scored.player_id == pid].iloc[0])


def test_invented_fraction_flagged(player_facts) -> None:
    f = player_facts("P0080")  # Shane Maddox
    md = "A plug-and-play floor spacer whose shooting should travel, even if his usage drops by a third."
    assert check_numbers(md, f) == ["a third"]


@pytest.mark.parametrize("phrase", ["twice as many threes", "half of his shots", "doubled his assists",
                                    "two-thirds of his points"])
def test_multiples_and_fractions_flagged(player_facts, phrase) -> None:
    assert check_numbers(phrase, player_facts("P0080")) != []


def test_spelled_numbers_must_round_from_data(player_facts) -> None:
    f = player_facts("P0080")  # 7.6 three-point attempts per game
    assert check_numbers("7.6 attempts per game, nearly eight a night.", f) == []
    assert check_numbers("He hits eleven threes a night.", f) == ["eleven"]
    assert check_numbers("He scored thirty-seven in the final.", f) == ["thirty-seven"]
    # Known looseness: a spelled-out integer passes if ANY input value rounds to it
    # ("thirty-four" matches his 33.5 minutes), so word checks catch most, not all.
    assert check_numbers("He scored thirty-four in the final.", f) == []


def test_zero_allowed_when_in_data(player_facts) -> None:
    f = player_facts("P0429")  # Tobias Zeller: 0 stars, 0 news mentions
    assert check_numbers("Zero stars and zero news coverage.", f) == []


def test_number_word_idioms_not_flagged(player_facts) -> None:
    md = ("A two-way guard who can play the four in a small-ball four look, shoots it from three, "
          "and is one of the best passers here. A triple-double threat with one season left.")
    assert check_numbers(md, player_facts("P0080")) == []


@pytest.mark.parametrize("sentence", [
    "He has done this in the Crossroads, and the NAIA multiplier of 0.74 is the steepest discount on this board.",
    "Sunbelt Valley conference strength of 0.45 is the weakest competition among this group.",
    "Highest Translation Score in this group at 94.2.",
    "D3 to D1 is the biggest jump on this board.",
    "He is the best shooter in the conference.",
])
def test_unsupported_comparisons_flagged(sentence) -> None:
    assert check_comparisons(sentence) == [sentence.rstrip(".")]


@pytest.mark.parametrize("sentence", [
    "Best shooting profile for his position: 100th percentile among D1 shooting guards.",
    "His 4.62 assist-to-turnover ratio is the top mark among D1 point guards at the 99th percentile.",
    "Shooting is the most portable skill in basketball.",
])
def test_d1_percentile_comparisons_allowed(sentence) -> None:
    assert check_comparisons(sentence) == []


def test_comparison_keeps_decimals_intact() -> None:
    out = check_comparisons("- **Best shooting profile on the board: 65.7% true shooting (63.0% adjusted).**")
    assert out == ["Best shooting profile on the board: 65.7% true shooting (63.0% adjusted)"]


def test_retry_feedback_includes_comparisons(row, tmp_path, monkeypatch) -> None:
    calls = []

    def fake(f, feedback=None):
        calls.append(feedback)
        if feedback is None:
            return _good_llm_md(f) + "\n\nHe is the best guard on this board."
        return _good_llm_md(f)

    monkeypatch.setattr(report, "llm_report", fake)
    out = generate_report(row, db_path=tmp_path / "r.duckdb", use_llm=True)
    assert "best guard on this board" in calls[1]
    assert out.unsupported_claims == [] and len(calls) == 2


def test_claims_survive_cache_round_trip(row, tmp_path, monkeypatch) -> None:
    def always_compares(f, feedback=None):
        return _good_llm_md(f) + "\n\nHe is the best guard on this board."

    monkeypatch.setattr(report, "llm_report", always_compares)
    db = tmp_path / "r.duckdb"
    first = generate_report(row, db_path=db, use_llm=True)
    cached = generate_report(row, db_path=db, use_llm=True)
    assert cached.cached and cached.unsupported_claims == first.unsupported_claims != []
