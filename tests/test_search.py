"""Phase 3: DuckDB search backend filters, gate, emphasis, and JSON safety."""

from __future__ import annotations

import json

import pytest

from scout.config import SCORING_CONFIG
from scout.query_parser import SearchFilters
from scout.search import DuckDBSearchBackend
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def backend() -> DuckDBSearchBackend:
    players, _ = generate_players(seed=42)
    return DuckDBSearchBackend(players=players)


def test_filters_respected(backend) -> None:
    f = SearchFilters(division=["D2"], position=["PG"], height_max_in=74)
    res = backend.search(f, limit=50)["results"]
    assert res
    assert all(r["division"] == "D2" and r["position"] == "PG" and r["height_in"] <= 74 for r in res)


def test_gate_applies_to_underrated_only(backend) -> None:
    gate = SCORING_CONFIG["min_translation_to_rank"]
    under = backend.search(SearchFilters(), limit=600)
    assert all(r["translation_score"] >= gate for r in under["results"])
    assert under["excluded_by_gate"] > 0
    assert under["total_matches"] == len(under["results"]) + under["excluded_by_gate"]
    trans = backend.search(SearchFilters(sort_by="translation"), limit=600)
    assert trans["excluded_by_gate"] == 0 and len(trans["results"]) == 600


def test_gate_can_be_disabled(backend, monkeypatch) -> None:
    monkeypatch.setitem(SCORING_CONFIG, "min_translation_to_rank", None)
    assert backend.search(SearchFilters(), limit=600)["excluded_by_gate"] == 0


def test_results_sorted_by_rank(backend) -> None:
    res = backend.search(SearchFilters(stat_emphasis=["passing"]), limit=30)["results"]
    ranks = [r["rank_score"] for r in res]
    assert ranks == sorted(ranks, reverse=True)
    assert all(r["emphasis_score"] is not None for r in res)


def test_emphasis_changes_order(backend) -> None:
    base = backend.search(SearchFilters(position=["C"]), limit=10)["results"]
    reb = backend.search(SearchFilters(position=["C"], stat_emphasis=["rebounding"]), limit=10)["results"]
    assert [r["player_id"] for r in base] != [r["player_id"] for r in reb]


def test_text_match(backend) -> None:
    school = backend.scored.iloc[0]["school"]
    res = backend.search(SearchFilters(text_match=school.split()[0], sort_by="translation"), limit=50)
    assert res["results"] and all(school.split()[0].lower() in (r["school"] + r["name"]).lower()
                                  for r in res["results"])


def test_results_are_json_serializable(backend) -> None:
    json.dumps(backend.search(SearchFilters(), limit=5))


def test_get_player(backend) -> None:
    pid = backend.scored.iloc[0]["player_id"]
    assert backend.get_player(pid)["player_id"] == pid
    assert backend.get_player("NOPE") is None
