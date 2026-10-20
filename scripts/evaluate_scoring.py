"""Check scoring against the planted ground truth.

Acceptance: >= 10 of 15 planted gems rank in the top 50 by Underrated Score.
Also reports volume decoys in the top 50 (they should mostly NOT be there).

Run:  python -m scripts.evaluate_scoring
"""

from __future__ import annotations

import pandas as pd

from scout.config import SCORING_CONFIG
from scout.db import fetch_planted, fetch_players, get_connection
from scout.scoring import explain_player, score_players

TOP_N = 50


def main() -> None:
    """Score everything, print the acceptance check and the top 10 with explanations."""
    with get_connection(read_only=True) as con:
        players, planted = fetch_players(con), fetch_planted(con)
    scored = score_players(players).merge(planted, on="player_id", how="left")
    scored["kind"] = scored["kind"].fillna("normal")
    scored = scored.sort_values("underrated_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    top = scored.head(TOP_N)

    gems_in = int((top["kind"] == "gem").sum())
    decoys_in = int((top["kind"] == "decoy").sum())
    print(f"Gems in top {TOP_N}:   {gems_in}/15  -> {'PASS' if gems_in >= 10 else 'FAIL'} (need >= 10)")
    print(f"Decoys in top {TOP_N}: {decoys_in}/10  (lower is better)")
    print(f"Top {TOP_N} by division: {top['division'].value_counts().to_dict()}")

    gate = SCORING_CONFIG["min_translation_to_rank"]
    if gate is not None:
        gated = scored[scored["translation_score"] >= gate].head(TOP_N)
        print(f"With ranking gate (Translation >= {gate:.0f}): gems {int((gated['kind'] == 'gem').sum())}/15, "
              f"decoys {int((gated['kind'] == 'decoy').sum())}/10")

    cols = ["rank", "player_id", "name", "division", "position", "kind",
            "translation_score", "attention_score", "underrated_score"]
    pd.set_option("display.width", 160)
    print("\nPlanted rows and their ranks:")
    print(scored.loc[scored["kind"] != "normal", cols].round(1).to_string(index=False))

    print(f"\n=== Top 10 by Underrated Score ===")
    for _, row in scored.head(10).iterrows():
        exp = explain_player(row)
        print(f"\n#{row['rank']} {row['name']} ({row['school']}, {row['division']} {row['position']}, "
              f"{row['height_in'] // 12}'{row['height_in'] % 12}\")  [{row['kind']}]")
        print(f"   Translation {row['translation_score']:.1f} | Attention {row['attention_score']:.1f} "
              f"| Underrated {row['underrated_score']:.1f}")
        for f in exp["top_factors"]:
            print(f"   + {f['text']}")
        print(f"   - {exp['top_concern']['text']}")
        for note in exp["notes"]:
            print(f"   . {note}")


if __name__ == "__main__":
    main()
