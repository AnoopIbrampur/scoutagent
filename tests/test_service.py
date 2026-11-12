"""One-process mode: the app builds its own data on first boot, and the in-process
service returns the same answers as the HTTP API."""

from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from scout.service import ScoutService, UnknownPlayer, ensure_database


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    db = tmp_path_factory.mktemp("fresh") / "scout.duckdb"
    assert not db.exists()
    assert ensure_database(db) is True  # missing -> built
    return db


def test_first_boot_builds_the_dataset(built) -> None:
    with duckdb.connect(str(built), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM players").fetchone()[0] == 600
        assert con.execute("SELECT count(*) FROM synergy_playtypes_offense").fetchone()[0] > 0
    assert ensure_database(built) is False  # already there -> left alone


def test_service_matches_http_api(built, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    svc = ScoutService(db_path=built)
    q = "D2 point guard under 6'3\" who can pass"
    local = svc.search(q, limit=10)
    with TestClient(create_app(db_path=built)) as c:
        remote = c.post("/search", json={"query": q, "limit": 10}).json()
        pid = local["results"][0]["player_id"]
        assert c.get(f"/players/{pid}").json()["summary"] == svc.player(pid)["summary"]
    assert [r["player_id"] for r in local["results"]] == [r["player_id"] for r in remote["results"]]
    rep = svc.report(pid)
    assert rep["method"] == "template" and rep["flagged_numbers"] == []


def test_unknown_player(built) -> None:
    with pytest.raises(UnknownPlayer):
        ScoutService(db_path=built).player("NOPE")


def test_concurrent_searches_are_safe(built) -> None:
    """Hosted visitors share one service; simultaneous searches must not collide."""
    from concurrent.futures import ThreadPoolExecutor
    svc = ScoutService(db_path=built)
    queries = ["D2 point guard who can pass", "NAIA center", "shooting wings from small schools",
               "D3 guard under 6'1", "lockdown defender in D2", "most D1-ready bigs"] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: svc.search(q, limit=5), queries))
    assert len(results) == len(queries) and all("results" in r for r in results)
    # Same query, same answer, regardless of what ran alongside it.
    assert results[0]["results"] == results[6]["results"]


def test_player_and_search_carry_the_agency_view(built) -> None:
    svc = ScoutService(db_path=built)
    res = svc.search("D2 point guard who can pass", limit=5)
    assert all(r["call"] in ("Sign", "Watch", "Pass") for r in res["results"])
    ag = svc.player(res["results"][0]["player_id"])["agency"]
    assert ag["call"] in ("Sign", "Watch", "Pass") and len(ag["fits"]) == 5 and "leverage" in ag
    rep = svc.report(res["results"][0]["player_id"], force=True)
    assert "## Best-Fit Destinations" in rep["markdown"] and rep["flagged_numbers"] == []
