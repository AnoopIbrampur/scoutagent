"""Phase 4: report fact check, template fallback, LLM retry, and caching."""

from __future__ import annotations

import pandas as pd
import pytest

import scout.report as report
from scout.llm import LLMError
from scout.report import (
    SECTIONS, build_report_input, check_numbers, generate_report, missing_sections, template_report,
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
