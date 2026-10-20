"""Parse a folder of Synergy "Player Details > Overall" PDF exports and check them.

Run:  python -m scripts.import_synergy_pdfs path/to/folder          (parse + validate + summary)
      python -m scripts.import_synergy_pdfs path/to/folder --load   (also store in DuckDB)

Real exports are licensed data. Keep the PDFs outside the repo (or under data/,
which is gitignored). --load writes to synergy_real_offense / synergy_real_defense,
separate from the synthetic tables, because real players also need a box-score
source before they can be scored.
"""

from __future__ import annotations

import argparse

from scout.db import get_connection, load_synergy
from scout.synergy_pdf import SynergyPdfSource


def main() -> None:
    """Parse, validate, summarize, and optionally load."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--load", action="store_true", help="store parsed tables in DuckDB")
    args = ap.parse_args()

    offense, defense, problems = SynergyPdfSource(args.folder).load()
    players = sorted(set(offense["player_id"]) | set(defense["player_id"]))
    print(f"Parsed {len(players)} player(s): {len(offense)} offense rows, {len(defense)} defense rows")
    for pid in players:
        o = offense[(offense.player_id == pid) & (offense.situation == "All Possessions")]
        d = defense[(defense.player_id == pid) & (defense.situation == "All Possessions")]
        line = f"  {pid}"
        if not o.empty:
            r = o.iloc[0]
            line += f" | OFF {r.poss:.0f} poss, {r.ppp:.3f} PPP ({r.rating}), SSM {r.ssm:+.2f}, TO% {100 * r.to_pct:.1f}%"
        if not d.empty:
            r = d.iloc[0]
            line += f" | DEF {r.poss:.0f} poss, {r.ppp:.3f} PPP allowed ({r.rating})"
        print(line)
    print("Rule check:", "all consistent" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print("  -", p)

    if args.load:
        with get_connection() as con:
            load_synergy(con, offense, defense, prefix="synergy_real")
        print("Stored in synergy_real_offense / synergy_real_defense.")


if __name__ == "__main__":
    main()
