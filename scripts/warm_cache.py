"""Pre-generate and cache reports for the players most likely to come up in a demo.

Run before presenting so report clicks are instant and survive a dead network:
    python -m scripts.warm_cache            (top 25 by Underrated, gate applied)
    python -m scripts.warm_cache 50
"""

from __future__ import annotations

import sys

from scout.query_parser import SearchFilters
from scout.report import generate_report
from scout.search import DuckDBSearchBackend


def main(n: int = 25) -> None:
    """Generate (or confirm cached) reports for the top-n underrated players."""
    backend = DuckDBSearchBackend()
    results = backend.search(SearchFilters(), limit=n)["results"]
    for r in results:
        rep = generate_report(backend.get_player(r["player_id"]), playtypes=backend.playtype_summary(r["player_id"]))
        status = "cached" if rep.cached else f"new ({rep.method})"
        flag = f"  FLAGGED {rep.flagged_numbers}" if rep.flagged_numbers else ""
        print(f"{r['player_id']} {r['name']:20s} {status}{flag}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
