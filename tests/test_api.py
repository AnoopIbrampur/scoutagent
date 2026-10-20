"""Phase 4: FastAPI endpoints end to end (offline: keyword parser + template reports)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from scout.search import DuckDBSearchBackend
from scout.synergy import generate_synergy
from scripts.generate_seed_data import generate_players


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module) -> TestClient:
    players, _ = generate_players(seed=42)
    app = create_app(backend=DuckDBSearchBackend(players=players, synergy=generate_synergy(players, seed=42)),
                     db_path=tmp_path_factory.mktemp("db") / "t.duckdb")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.setenv("ANTHROPIC_API_KEY", "")  # force offline paths regardless of local .env
    yield mp
    mp.undo()


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["players"] == 600 and body["llm_enabled"] is False


def test_search_demo_query(client) -> None:
    body = client.post("/search", json={"query": "D2 point guard under 6'3\" strong passer"}).json()
    assert body["parsed"]["method"] == "keyword"
    assert "Division: D2" in body["parsed"]["chips"]
    assert body["results"] and all(r["division"] == "D2" for r in body["results"])
    assert {"translation_score", "underrated_score", "top_factor", "explanation"} <= set(body["results"][0])


def test_search_include_gated(client) -> None:
    gated = client.post("/search", json={"query": "NAIA center"}).json()
    full = client.post("/search", json={"query": "NAIA center", "include_gated": True, "limit": 100}).json()
    assert full["excluded_by_gate"] == 0 and len(full["results"]) >= len(gated["results"])


def test_search_validation(client) -> None:
    assert client.post("/search", json={}).status_code == 422


def test_player_and_report(client) -> None:
    pid = client.post("/search", json={"query": "D2 PG"}).json()["results"][0]["player_id"]
    p = client.get(f"/players/{pid}").json()
    assert p["summary"]["player_id"] == pid and p["percentiles_vs_d1_position"]
    rep = client.post(f"/players/{pid}/report").json()
    assert rep["method"] == "template" and rep["flagged_numbers"] == [] and not rep["cached"]
    assert client.post(f"/players/{pid}/report").json()["cached"]
    assert not client.post(f"/players/{pid}/report", params={"force": True}).json()["cached"]


def test_unknown_player_404(client) -> None:
    assert client.get("/players/NOPE").status_code == 404
    assert client.post("/players/NOPE/report").status_code == 404


def test_player_includes_synergy_splits(client) -> None:
    syn = client.get("/players/P0001").json()["synergy"]
    assert [len(syn["offense"]), len(syn["defense"])] == [12, 12]
    assert syn["offense"][0]["situation"] == "All Possessions" and "ssm" in syn["offense"][0]
