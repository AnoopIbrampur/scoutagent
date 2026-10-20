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
from scout.synergy import calibrate_ratings, rating_mismatches
from scout.synergy_pdf import SynergyPdfSource


def main() -> None:
    """Parse, validate, summarize, and optionally load."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--load", action="store_true", help="store parsed tables in DuckDB")
    args = ap.parse_args()

    tables, problems, _ = SynergyPdfSource(args.folder).load_all()
    offense, defense = tables["offense"], tables["defense"]
    players = sorted(set(offense["player_id"]) | set(defense["player_id"]))
    print(f"Parsed {len(players)} player(s): {len(offense)} offense / {len(defense)} defense rows, "
          f"{len(tables['playtypes_offense'])} / {len(tables['playtypes_defense'])} play-type rows")
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

    nonempty = [t for t in tables.values() if not t.empty]
    if nonempty:
        print("\nRATING cutoffs implied by these exports (each cutoff must be > 'above' and <= 'at most'):")
        print(calibrate_ratings(nonempty).to_string(index=False))
        mismatches = [m for t in nonempty for m in rating_mismatches(t)]
        print("Ratings that disagree with config.SYNERGY_RATING_CUTOFFS:",
              "none" if not mismatches else "")
        for m in mismatches:
            print("  -", m)

    if args.load:
        with get_connection() as con:
            load_synergy(con, offense, defense, prefix="synergy_real",
                         playtypes={k: tables[k] for k in ("playtypes_offense", "playtypes_defense")})
        print("Stored in synergy_real_* tables. To score these players, add a roster: "
              "python -m scripts.load_real_players --help")


if __name__ == "__main__":
    main()
