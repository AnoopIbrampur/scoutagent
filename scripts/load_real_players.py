"""Turn real Synergy exports + a roster CSV into scoreable players.

Step 1, make a roster template pre-filled from the exports:
    python -m scripts.load_real_players --pdfs ~/synergy_exports --template data/roster.csv
Step 2, fill in the blank cells (height, position, division, minutes, rebounds, ...).
Step 3, validate, store, and preview the rankings:
    python -m scripts.load_real_players --pdfs ~/synergy_exports --roster data/roster.csv
Step 4, run the app on real players:
    SCOUT_DATASET=real uvicorn api.main:app --port 8000

Real players are compared against the synthetic D1 pool until real D1 data is
loaded; the UI and reports say so. Keep PDFs and roster files out of git
(data/ and *.pdf are gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scout.config import PROJECT_ROOT
from scout.data_sources import DataSourceError
from scout.db import get_connection, load_real_players, load_synergy
from scout.roster import RosterCsvSource, roster_template
from scout.search import DuckDBSearchBackend
from scout.synergy_pdf import SynergyPdfSource


def _warn_if_tracked(path: Path) -> None:
    """Real data should live under data/ (gitignored) or outside the repo."""
    path = path.resolve()
    if PROJECT_ROOT in path.parents and (PROJECT_ROOT / "data") not in path.parents:
        print(f"WARNING: {path} is inside the repo but not under data/, so git could pick it up.")


def make_template(pdfs: Path, out: Path) -> None:
    """Write a roster CSV with one pre-filled row per exported player."""
    tables, problems, exports = SynergyPdfSource(pdfs).load_all()
    template = roster_template(exports)
    if template.empty:
        raise SystemExit("No Overall offense exports found; the template needs those.")
    out.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out, index=False)
    _warn_if_tracked(out)
    blank = [c for c in template.columns if (template[c] == "").all()]
    print(f"Wrote {len(template)} player(s) to {out}")
    print("Pre-filled from Synergy: name, school, games, points_pg, fg_pct, ft_att_pg, ft_pct, turnovers_pg")
    print(f"Fill in: {', '.join(blank)}")
    print("Optional: conference_strength (default 0.5) and the attention columns "
          "(recruiting_stars, social_followers, news_mentions_90d); blank attention = unknown.")
    if problems:
        print("Export problems:", *problems, sep="\n  - ")


def load(pdfs: Path, roster: Path) -> None:
    """Validate roster + exports, store them, and print the real-player rankings."""
    tables, problems, _ = SynergyPdfSource(pdfs).load_all()
    try:
        players = RosterCsvSource(roster).fetch_players()
    except DataSourceError as e:
        raise SystemExit(f"Roster problem: {e}")
    with_syn = set(tables["offense"]["player_id"])
    no_syn = sorted(set(players["player_id"]) - with_syn)
    not_in_roster = sorted(with_syn - set(players["player_id"]))
    with get_connection() as con:
        n = load_real_players(con, players)
        load_synergy(con, tables["offense"], tables["defense"], prefix="synergy_real",
                     playtypes={k: tables[k] for k in ("playtypes_offense", "playtypes_defense")})
    print(f"Stored {n} real player(s) and their Synergy tables.")
    if no_syn:
        print("No Synergy export (scored on roster stats only):", ", ".join(no_syn))
    if not_in_roster:
        print("Exports with no roster row (ignored until added):", ", ".join(not_in_roster))
    if problems:
        print("Export problems:", *problems, sep="\n  - ")

    scored = DuckDBSearchBackend(dataset="real").scored.sort_values("translation_score", ascending=False)
    print("\nReal players vs the synthetic D1 baseline:")
    for _, r in scored.iterrows():
        u = "n/a" if r["underrated_score"] != r["underrated_score"] else f"{r['underrated_score']:.1f}"
        print(f"  {r['name']:24s} {r['division']:4s} {r['position']:2s} Translation {r['translation_score']:5.1f}  "
              f"Underrated {u}")
    print("\nRun the app on them:  SCOUT_DATASET=real uvicorn api.main:app --port 8000")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdfs", type=Path, required=True, help="folder of Synergy PDF exports")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--template", type=Path, help="write a pre-filled roster CSV here")
    group.add_argument("--roster", type=Path, help="filled-in roster CSV to load")
    args = ap.parse_args()
    if args.template:
        make_template(args.pdfs, args.template)
    else:
        load(args.pdfs, args.roster)


if __name__ == "__main__":
    main()
