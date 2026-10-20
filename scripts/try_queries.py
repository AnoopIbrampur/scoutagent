"""Run sample queries end to end: parse -> search -> print. Uses Claude if a key is set.

Run:  python -m scripts.try_queries                 (5 built-in queries)
      python -m scripts.try_queries "your query"    (one custom query)
"""

from __future__ import annotations

import sys

from scout.query_parser import parse_query
from scout.search import DuckDBSearchBackend

SAMPLE_QUERIES = [
    "Find a D2 point guard who'd translate to D1, strong passer, under 6'3\"",
    "Best shooting wings from small schools",
    "NAIA or D3 center who protects the rim, at least 6'9",
    "most D1-ready bigs in division ii, rebounding",
    "need a guard lol... like 6-1 or shorter?? d3 maybe naia, takes care of the ball & can shoot it",
]


def main(queries: list[str]) -> None:
    """Parse and search each query, printing chips, SQL, and the top 5."""
    backend = DuckDBSearchBackend()
    for q in queries:
        parsed = parse_query(q, known_schools=backend.schools)
        res = backend.search(parsed.filters, limit=5)
        print(f"\nQ: {q}")
        print(f"   parsed by: {parsed.method}   {' | '.join(parsed.chips())}")
        for note in parsed.notes:
            print(f"   note: {note}")
        print(f"   SQL WHERE {res['where_sql']}  ->  {res['total_matches']} matches, "
              f"{res['excluded_by_gate']} hidden (Translation < {res['gate']})")
        for r in res["results"]:
            print(f"   {r['rank_score']:5.1f}  {r['name']:20s} {r['division']:4s} {r['position']:2s} "
                  f"{r['height']:6s} T={r['translation_score']:5.1f} U={r['underrated_score']:5.1f} | {r['top_factor']}")


if __name__ == "__main__":
    main(sys.argv[1:] or SAMPLE_QUERIES)
