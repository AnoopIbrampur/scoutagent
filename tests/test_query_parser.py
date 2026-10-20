"""Phase 3: LLM parse -> retry -> keyword fallback, and keyword parser coverage."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scout.llm import LLMError
from scout.query_parser import SearchFilters, keyword_parse, parse_query

GOOD = json.dumps({"division": ["D2"], "position": ["PG"], "height_max_in": 74, "height_min_in": None,
                   "stat_emphasis": ["passing"], "sort_by": "underrated", "text_match": None})
BAD_ENUM = json.dumps({"division": ["D9"], "position": [], "height_max_in": None, "height_min_in": None,
                       "stat_emphasis": [], "sort_by": "underrated", "text_match": None})


class FakeLLM:
    """Returns queued responses; raises if an item is an exception."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append(user)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_llm_success() -> None:
    llm = FakeLLM(GOOD)
    out = parse_query("D2 PG under 6'3 who can pass", llm=llm)
    assert out.method == "llm"
    assert out.filters.division == ["D2"] and out.filters.height_max_in == 74
    assert len(llm.calls) == 1


def test_llm_retry_then_success() -> None:
    llm = FakeLLM(BAD_ENUM, GOOD)
    out = parse_query("D2 PG", llm=llm)
    assert out.method == "llm_retry"
    assert "previous answer was invalid" in llm.calls[1]


def test_llm_invalid_twice_falls_back() -> None:
    out = parse_query("D2 point guard", llm=FakeLLM("not json", BAD_ENUM))
    assert out.method == "keyword"
    assert out.filters.division == ["D2"] and out.filters.position == ["PG"]
    assert any("invalid twice" in n for n in out.notes)


def test_llm_api_error_falls_back_without_retry() -> None:
    llm = FakeLLM(LLMError("Rate limited"))
    out = parse_query("NAIA center", llm=llm)
    assert out.method == "keyword" and len(llm.calls) == 1
    assert out.filters.division == ["NAIA"] and out.filters.position == ["C"]
    assert any("Rate limited" in n for n in out.notes)


def test_no_key_uses_keyword(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    out = parse_query("D3 wing")
    assert out.method == "keyword"


def test_filters_reject_inverted_heights() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(height_min_in=80, height_max_in=75)


@pytest.mark.parametrize("query, expected", [
    ("Find a D2 point guard who'd translate to D1, strong passer, under 6'3\"",
     dict(division=["D2"], position=["PG"], height_max_in=74, stat_emphasis=["passing"])),
    ("D1-ready wings from small schools", dict(division=["D2", "D3", "NAIA"], position=["SG", "SF"])),
    ("division iii center at least 6'9 who blocks shots",
     dict(division=["D3"], position=["C"], height_min_in=81, stat_emphasis=["rim_protection"])),
    ("6-1 or shorter guard who takes care of the ball",
     dict(position=["PG", "SG"], height_max_in=73, stat_emphasis=["ball_security"])),
    ("a 6'5 wing", dict(position=["SG", "SF"], height_min_in=76, height_max_in=78)),
    ("over 6’8” big man", dict(position=["C"], height_min_in=81)),
    ("most D1-ready bigs", dict(position=["PF", "C"], sort_by="translation", division=[])),
    ("who could play D1 from NAIA", dict(division=["NAIA"])),
])
def test_keyword_parser(query: str, expected: dict) -> None:
    f = keyword_parse(query)
    for key, value in expected.items():
        assert getattr(f, key) == value, (key, getattr(f, key))


def test_keyword_parser_empty_query() -> None:
    f = keyword_parse("")
    assert f == SearchFilters()


def test_chips_show_height_in_feet() -> None:
    out = parse_query("D2 PG under 6'3", llm=FakeLLM(GOOD))
    assert 'Height ≤ 6\'2"' in out.chips()


def test_keyword_parser_matches_known_school() -> None:
    f = keyword_parse("any guards at ravenna tech?", known_schools=["Ravenna Tech", "Ravenna State"])
    assert f.text_match == "Ravenna Tech" and f.position == ["PG", "SG"]


def test_llm_schema_matches_pydantic_model() -> None:
    from scout.query_parser import _json_schema
    schema = _json_schema()
    assert set(schema["properties"]) == set(SearchFilters.model_fields) == set(schema["required"])
