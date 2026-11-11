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
