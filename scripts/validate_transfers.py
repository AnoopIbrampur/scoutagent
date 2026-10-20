"""Check the scoring against players who actually moved up to D1.

    python -m scripts.validate_transfers --simulate                 # self-test on fake transfers
    python -m scripts.validate_transfers --transfers data/transfers.csv
    python -m scripts.validate_transfers --template data/transfers.csv   # blank file to fill in

Real transfer data describes real people: keep it under data/ (gitignored).
See scout/validation.py for exactly what each number means.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scout.data_sources import DataSourceError
from scout.validation import (
    MIN_POST_MINUTES, OUTCOMES, PREDICTORS, ROTATION_MPG, TRANSFER_COLUMNS, TRUE_FACTORS, calibrate_levels,
    load_transfers, predictive_check, rotation_hit_rates, score_transfers, simulate_transfers,
)
from scripts.generate_seed_data import generate_players


def report(transfers: pd.DataFrame, simulated: bool) -> None:
    """Print level calibration, predictive correlations, and rotation hit rates."""
    reference, _ = generate_players(seed=42)  # D1 percentile baseline (synthetic until real D1 data exists)
    scored = score_transfers(transfers, reference)
    counts = transfers["from_division"].value_counts().to_dict()
    print(f"{len(transfers)} players moved up to D1: {counts}\n")

    print(f"1. Are the level factors right?  (players with {MIN_POST_MINUTES}+ D1 minutes)")
    print("   ratio = actual D1 production / what the factors predicted. 1.0 means the factor is right.\n")
    print(f"   {'':5} {'n':>3}  {'config':>6}  {'ratio':>6}  {'95% CI':>13}  {'suggested factor (95% CI)':>27}"
          + ("  truth" if simulated else ""))
    for c in calibrate_levels(transfers):
        lo, hi = c.suggested_ci
        verdict = "" if c.n >= 15 else "  (too few players to trust)"
        truth = f"  {TRUE_FACTORS[c.division]:.2f}" if simulated else ""
        print(f"   {c.division:5} {c.n:>3}  {c.configured:>6.2f}  {c.median_ratio:>6.3f}  ({c.ci[0]:.3f}, {c.ci[1]:.3f})  "
              f"{c.suggested:>8.3f}  ({lo:.3f}, {hi:.3f}){truth}{verdict}")
    print("   Caveat: players who barely played at D1 are left out, which makes the factors look better than they are.\n")

    print("2. Does the pre-transfer score predict D1 success?  (Spearman rank correlation, 95% CI)")
    for outcome, label in OUTCOMES.items():
        print(f"\n   Outcome: {label}")
        for r in [r for r in predictive_check(scored) if r.outcome == outcome]:
            print(f"     {PREDICTORS[r.predictor]:34s} rho {r.rho:+.2f}  ({r.ci[0]:+.2f}, {r.ci[1]:+.2f})  n={r.n}")
    print("\n   Read minutes first: it's the coaches' verdict and doesn't use our formula. D1 production reuses")
    print("   the Translation formula, so Translation has a built-in edge there. If Translation doesn't beat")
    print("   raw points per game on minutes, the extra machinery isn't earning its keep yet.\n")

    print(f"3. Who became a D1 rotation player ({ROTATION_MPG:.0f}+ minutes per game)?")
    hits = rotation_hit_rates(scored)
    for third, r in hits.iterrows():
        print(f"   {third:13s} {r['rotation_rate']:>5.0%} of {int(r['players'])} players  (average Translation {r['avg_translation']:.1f})")
    if simulated:
        print("\nThis was the self-test on simulated transfers with known factors. The point is whether the")
        print("suggested factors land near the truth column. Real conclusions need --transfers with real data.")
    else:
        print("\nPercentiles use the synthetic D1 pool as the baseline, so Translation's scale is approximate;")
        print("the rank correlations and level ratios don't depend on that baseline.")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulate", action="store_true", help="self-test on simulated transfers with known factors")
    g.add_argument("--transfers", type=Path, help="CSV of real move-ups (see --template)")
    g.add_argument("--template", type=Path, help="write a blank transfers CSV with the right columns")
    args = ap.parse_args()
    if args.template:
        args.template.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=TRANSFER_COLUMNS).to_csv(args.template, index=False)
        print(f"Wrote {args.template} with {len(TRANSFER_COLUMNS)} columns. One row per player: his last small-school")
        print("season (pre_*) and his first D1 season (post_*). If he never played at D1, set post_games to 0")
        print("and leave the other post_ cells blank. Example: docs/transfers_example.csv")
        return
    if args.simulate:
        players, _ = generate_players(seed=42)
        report(simulate_transfers(players), simulated=True)
        return
    try:
        transfers = load_transfers(args.transfers)
    except DataSourceError as e:
        raise SystemExit(f"Transfers file problem: {e}")
    report(transfers, simulated=False)


if __name__ == "__main__":
    main()
