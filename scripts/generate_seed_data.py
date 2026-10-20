"""Generate a synthetic NCAA men's basketball dataset and load it into DuckDB.

Why synthetic: many stats sites prohibit scraping, and planted ground truth
lets us test whether the scoring actually finds overlooked players.

Model (kept simple on purpose):
  * Every player has a latent skill `r ~ N(0, 1)` measured *relative to their
    own league*. So a D3 star and a D1 star post similar raw numbers. That is
    realistic (weaker opponents inflate box scores) and it is exactly why
    scoring must apply a level adjustment.
  * Box-score rates are generated per 36 minutes from position-specific
    baselines, scaled by r, then converted to per-game by minutes played.
  * Points are *derived* from shot attempts and percentages, so true shooting
    can be reconstructed exactly from the stored per-game fields.
  * Attention (stars, followers, news) depends mostly on division and
    conference, and only weakly on skill. That gap is the whole pitch.

Planted rows (written to a separate table, never used by scoring):
  * 15 "hidden gems": non-D1, high skill, plus an efficiency edge (shooting,
    ball security) and suppressed attention.
  * 10 "volume decoys": non-D1, low attention, lots of shots but inefficient.
    A good Underrated Score should NOT reward these just for being unknown.

Run:  python -m scripts.generate_seed_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scout.config import DATA_DIR, SEED_PLANTED_CSV, SEED_PLAYERS_CSV
from scout.data_sources import SeedDataSource
from scout.db import get_connection, load_players

SEED = 42

# (conference name, strength 0-1, number of schools). Strength is relative
# within a division. Rosters: D1 12, D2 11, D3 10, NAIA 10 -> 360/110/80/50 = 600.
CONFERENCES: dict[str, list[tuple[str, float, int]]] = {
    "D1": [("Atlantic Premier", 0.95, 5), ("Great Plains", 0.85, 5), ("Pacific Coast", 0.75, 5),
           ("Mid-Valley", 0.55, 5), ("Northern Lakes", 0.40, 5), ("Southern Coastal", 0.25, 5)],
    "D2": [("Heartland", 0.80, 3), ("Bluegrass", 0.65, 3), ("Sunbelt Valley", 0.45, 2), ("Frontier", 0.30, 2)],
    "D3": [("Commonwealth", 0.70, 2), ("Lakeshore", 0.55, 2), ("Old Colony", 0.40, 2), ("Prairie", 0.25, 2)],
    "NAIA": [("Crossroads", 0.70, 2), ("Red River", 0.45, 2), ("High Desert", 0.30, 1)],
}
ROSTER_SIZE = {"D1": 12, "D2": 11, "D3": 10, "NAIA": 10}
SCHOOL_PATTERNS = {
    "D1": ["{} State", "University of {}"],
    "D2": ["{} University", "{} State"],
    "D3": ["{} College"],
    "NAIA": ["{} Christian", "{} Tech"],
}
# Invented-sounding place names; any match to a real school is coincidental.
PLACES = [
    "Ashford", "Briarwood", "Calder", "Dunmore", "Easton Falls", "Fairhaven", "Glenrock",
    "Halston", "Ironwood", "Juniper Bay", "Kestrel", "Larkspur", "Millbrook", "Northgate",
    "Oakhurst", "Pinecrest", "Quarry Hill", "Ravenna", "Stonebridge", "Thornbury", "Upland",
    "Valecrest", "Yarrow", "Bellmere", "Corbin", "Elkridge", "Foxhall", "Granger", "Hollis",
    "Ivydale", "Kingsford", "Lowell Ridge", "Marlow", "Nettleton", "Orrin", "Pellham",
    "Redfield", "Sutter", "Tamsin", "Ulster Ridge", "Verona Hills", "Wexley", "Alder",
    "Birchmont", "Coldwater", "Emberly", "Fenwick", "Garrison", "Hartwell", "Brookhaven",
    "Cedar Point", "Silver Lake", "Maple Ridge", "Harrow", "Linden",
]
FIRST_NAMES = [
    "Jalen", "Marcus", "Tyrese", "Devin", "Cole", "Andre", "Isaiah", "Malik", "Trey", "Jordan",
    "Caleb", "Darius", "Elijah", "Xavier", "Noah", "Miles", "Jaylen", "Cameron", "Owen", "Bryce",
    "Kendall", "Terrance", "Luka", "Mateo", "Quentin", "Reggie", "Silas", "Tobias", "Wes", "Zion",
    "Amari", "Brandon", "Corey", "Dante", "Emeka", "Felix", "Grant", "Hassan", "Ian", "Jamal",
    "Kofi", "Landon", "Nate", "Omar", "Preston", "Rashad", "Shane", "Tariq", "Victor", "Wyatt",
]
LAST_NAMES = [
    "Abbott", "Barnes", "Calloway", "Dawson", "Ellison", "Fitzgerald", "Garrett", "Holloway",
    "Ingram", "Jefferson", "Kincaid", "Lawson", "Mercer", "Nwosu", "Okafor", "Pruitt", "Quinlan",
    "Ramsey", "Sutton", "Thibodeaux", "Underwood", "Vance", "Whitaker", "Yates", "Ashby",
    "Brennan", "Crowder", "Delaney", "Easley", "Fowler", "Greer", "Hardaway", "Iverson", "Jett",
    "Keane", "Langley", "Maddox", "Norwood", "Osei", "Pickett", "Rourke", "Stallworth", "Tolliver",
    "Vasquez", "Wainwright", "Adeyemi", "Boateng", "Castellanos", "Dupree", "Everett", "Fairley",
    "Gibbs", "Hensley", "Jameson", "Kowalski", "Lockhart", "Monroe", "Nash", "Ortega", "Pena",
    "Redd", "Sampson", "Tate", "Upshaw", "Vaughn", "Winslow", "Young", "Zeller", "Bishop", "Cruz",
]

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
# Position baselines (per 36 minutes, average D1 rotation player).
HEIGHT_MEAN = {"PG": 73.5, "SG": 76.0, "SF": 78.0, "PF": 80.0, "C": 82.5}
DIVISION_HEIGHT_SHIFT = {"D1": 0.0, "D2": -0.7, "D3": -1.2, "NAIA": -1.0}
FGA36 = {"PG": 12.5, "SG": 13.5, "SF": 12.5, "PF": 11.0, "C": 10.0}
THREE_SHARE = {"PG": 0.42, "SG": 0.45, "SF": 0.36, "PF": 0.20, "C": 0.04}
TWO_PCT = {"PG": 0.47, "SG": 0.48, "SF": 0.49, "PF": 0.52, "C": 0.57}
FT_RATE = {"PG": 0.30, "SG": 0.28, "SF": 0.30, "PF": 0.34, "C": 0.40}
FT_PCT = {"PG": 0.76, "SG": 0.77, "SF": 0.73, "PF": 0.69, "C": 0.64}
REB36 = {"PG": 4.2, "SG": 4.8, "SF": 6.3, "PF": 8.4, "C": 10.0}
AST36 = {"PG": 6.3, "SG": 3.2, "SF": 2.6, "PF": 1.9, "C": 1.5}
STL36 = {"PG": 1.6, "SG": 1.4, "SF": 1.3, "PF": 1.0, "C": 0.8}
BLK36 = {"PG": 0.2, "SG": 0.3, "SF": 0.6, "PF": 1.1, "C": 2.0}
TOV36 = {"PG": 2.9, "SG": 2.1, "SF": 1.9, "PF": 1.9, "C": 2.1}

# Attention baselines by division: log-followers and log news rate.
LOG_FOLLOWERS = {"D1": 8.2, "D2": 6.4, "D3": 5.6, "NAIA": 5.9}
LOG_NEWS = {"D1": 1.6, "D2": 0.2, "D3": -0.5, "NAIA": -0.2}
STARS_BASE = {"D1": (2.2, 0.7), "D2": (0.6, 0.3), "D3": (0.2, 0.2), "NAIA": (0.4, 0.25)}

# Planted ground truth. Two D2 PGs are forced to <= 6'2" so the demo query
# "D2 point guard ... under 6'3"" has real answers.
GEM_PLAN = [
    ("D2", "PG"), ("D2", "PG"), ("D2", "PG"), ("D2", "SG"), ("D2", "SF"), ("D2", "PF"),
    ("D2", "C"), ("D2", "SF"), ("NAIA", "PG"), ("NAIA", "SG"), ("NAIA", "PF"), ("NAIA", "C"),
    ("D3", "PG"), ("D3", "SF"), ("D3", "C"),
]
DECOY_PLAN = [
    ("D2", "SG"), ("D2", "PG"), ("D2", "SF"), ("D3", "SG"), ("D3", "PG"),
    ("D3", "PF"), ("NAIA", "SG"), ("NAIA", "SF"), ("NAIA", "PG"), ("D2", "C"),
]


def build_schools(rng: np.random.Generator) -> pd.DataFrame:
    """Create fake schools with division, conference, and conference strength."""
    places = list(rng.permutation(PLACES))
    rows = []
    for division, confs in CONFERENCES.items():
        for conf, strength, n_schools in confs:
            for _ in range(n_schools):
                pattern = SCHOOL_PATTERNS[division][int(rng.integers(len(SCHOOL_PATTERNS[division])))]
                rows.append({"school": pattern.format(places.pop()), "division": division,
                             "conference": conf, "conference_strength": strength})
    return pd.DataFrame(rows)


def _unique_names(rng: np.random.Generator, n: int) -> list[str]:
    """Draw n distinct fake full names."""
    names: set[str] = set()
    while len(names) < n:
        names.add(f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}")
    return list(rng.permutation(sorted(names)))


def simulate_player(rng: np.random.Generator, division: str, position: str, kind: str = "normal") -> dict:
    """Simulate one season line. `kind` is "normal", "gem", or "decoy"."""
    gem, decoy = kind == "gem", kind == "decoy"
    r = rng.uniform(2.2, 2.9) if gem else rng.uniform(1.0, 1.6) if decoy else rng.normal()

    height = HEIGHT_MEAN[position] + DIVISION_HEIGHT_SHIFT[division] + rng.normal(0, 1.6)
    height = int(round(np.clip(height, 68, 87)))
    weight = int(round(190 + 5.5 * (height - 76) + rng.normal(0, 10)))

    if gem:
        minutes = rng.uniform(30, 35)
    elif decoy:
        minutes = rng.uniform(32, 36)
    else:
        minutes = float(np.clip(20 + 6 * r + rng.normal(0, 3), 6, 37))
    scale = minutes / 36.0

    # Shooting: attempts and makes, then derive points.
    fga36 = FGA36[position] * np.exp(0.12 * r + rng.normal(0, 0.12)) * (1.35 if decoy else 1.0)
    three_share = float(np.clip(THREE_SHARE[position] * np.exp(rng.normal(0, 0.25)), 0.0, 0.65))
    two_pct = float(np.clip(TWO_PCT[position] + 0.02 * r + rng.normal(0, 0.03)
                            + (0.04 if gem else 0) - (0.06 if decoy else 0), 0.35, 0.70))
    three_pct = float(np.clip(0.34 + 0.015 * r + rng.normal(0, 0.035)
                              + (0.04 if gem else 0) - (0.05 if decoy else 0), 0.20, 0.48))
    ft_rate = FT_RATE[position] * np.exp(rng.normal(0, 0.2))
    ft_pct = float(np.clip(FT_PCT[position] + 0.02 * r + rng.normal(0, 0.05)
                           + (0.04 if gem else 0), 0.45, 0.92))

    fga = fga36 * scale
    tpa = fga * three_share
    two_pa = fga - tpa
    fta = fga * ft_rate
    points = 2 * two_pa * two_pct + 3 * tpa * three_pct + fta * ft_pct
    fg_pct = (two_pa * two_pct + tpa * three_pct) / fga

    def rate(base: dict[str, float], skill_coef: float, noise: float) -> float:
        return base[position] * np.exp(skill_coef * r + rng.normal(0, noise)) * scale

    assists = rate(AST36, 0.15, 0.18)
    turnovers = rate(TOV36, 0.04, 0.18) * (0.70 if gem else 1.40 if decoy else 1.0)

    # Attention. Gems and decoys are deliberately under the radar.
    if gem or decoy:
        stars = int(rng.choice([0, 0, 1, 1, 2] if decoy else [0, 1, 1]))
        followers = int(np.exp(rng.normal(LOG_FOLLOWERS[division] - 0.6, 0.4)))
        news = int(rng.poisson(0.6))
    else:
        mu, coef = STARS_BASE[division]
        stars = int(np.clip(round(mu + coef * r + rng.normal(0, 0.7)), 0, 5))
        followers = int(np.exp(LOG_FOLLOWERS[division] + 0.45 * r + rng.normal(0, 0.7)))
        news = int(rng.poisson(np.exp(LOG_NEWS[division] + 0.5 * r)))

    return {
        "division": division, "position": position, "height_in": height, "weight_lb": weight,
        "class_year": str(rng.choice(["SO", "JR", "SR"] if gem else ["FR", "SO", "JR", "SR"])),
        "games": int(rng.integers(26, 35) if (gem or decoy) else rng.integers(18, 35)),
        "minutes_pg": round(minutes, 1),
        "points_pg": round(points, 1),
        "rebounds_pg": round(rate(REB36, 0.12, 0.15), 1),
        "assists_pg": round(assists, 1),
        "steals_pg": round(rate(STL36, 0.12, 0.20), 1),
        "blocks_pg": round(rate(BLK36, 0.15, 0.30), 1),
        "turnovers_pg": round(turnovers, 1),
        "fg_pct": round(fg_pct, 3),
        "three_pct": round(three_pct, 3),
        "three_att_pg": round(tpa, 1),
        "ft_pct": round(ft_pct, 3),
        "ft_att_pg": round(fta, 1),
        "recruiting_stars": stars,
        "social_followers": followers,
        "news_mentions_90d": news,
        "_kind": kind,
    }


def generate_players(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, planted). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    schools = build_schools(rng)

    # Fill every roster slot with a random position, then overwrite some
    # non-D1 slots with planted gems/decoys of the requested division+position.
    slots = []
    for school in schools.itertuples(index=False):
        for _ in range(ROSTER_SIZE[school.division]):
            slots.append({"school": school.school, "division": school.division,
                          "conference": school.conference,
                          "conference_strength": school.conference_strength,
                          "position": str(rng.choice(POSITIONS)), "kind": "normal"})
    slots_df = pd.DataFrame(slots)

    for kind, plan in (("gem", GEM_PLAN), ("decoy", DECOY_PLAN)):
        for division, position in plan:
            # At most one gem per school, so no single school looks rigged.
            used = set(slots_df.loc[slots_df["kind"] == "gem", "school"]) if kind == "gem" else set()
            open_idx = slots_df.index[(slots_df["division"] == division) & (slots_df["kind"] == "normal")
                                      & ~slots_df["school"].isin(used)]
            # Gems prefer stronger conferences (credible competition); decoys anywhere.
            weights = slots_df.loc[open_idx, "conference_strength"].to_numpy() ** (2 if kind == "gem" else 0)
            i = int(rng.choice(open_idx, p=weights / weights.sum()))
            slots_df.loc[i, ["position", "kind"]] = [position, kind]

    rows = []
    for slot in slots_df.itertuples(index=False):
        p = simulate_player(rng, slot.division, slot.position, slot.kind)
        p.update(school=slot.school, conference=slot.conference,
                 conference_strength=slot.conference_strength)
        rows.append(p)
    players = pd.DataFrame(rows)

    # Force two D2 PG gems to be <= 6'2" for the headline demo query.
    d2_pg_gems = players.index[(players["_kind"] == "gem") & (players["division"] == "D2")
                               & (players["position"] == "PG")][:2]
    players.loc[d2_pg_gems, "height_in"] = [73, 74]

    # Shuffle so planted rows don't cluster at the end, then assign IDs and names.
    players = players.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    players.insert(0, "player_id", [f"P{i:04d}" for i in range(1, len(players) + 1)])
    players.insert(1, "name", _unique_names(rng, len(players)))

    planted = players.loc[players["_kind"] != "normal", ["player_id", "_kind"]].rename(columns={"_kind": "kind"})
    planted = planted.reset_index(drop=True)
    players = players.drop(columns=["_kind"])
    return players, planted


def print_summary(players: pd.DataFrame, planted: pd.DataFrame) -> None:
    """Print sanity-check tables for eyeballing the generated data."""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== Players by division ===")
    counts = players["division"].value_counts().reindex(["D1", "D2", "D3", "NAIA"])
    print(pd.DataFrame({"players": counts, "share": (counts / len(players)).round(3)}))

    print("\n=== Per-game averages by position (all divisions) ===")
    cols = ["height_in", "minutes_pg", "points_pg", "rebounds_pg", "assists_pg",
            "blocks_pg", "turnovers_pg", "fg_pct", "three_pct", "ft_pct"]
    by_pos = players.groupby("position")[cols].mean().round(2).reindex(POSITIONS)
    by_pos.insert(0, "n", players["position"].value_counts().reindex(POSITIONS))
    print(by_pos)

    print("\n=== Production vs attention by division ===")
    agg = players.groupby("division").agg(
        points_pg=("points_pg", "mean"), fg_pct=("fg_pct", "mean"),
        stars=("recruiting_stars", "mean"), followers_median=("social_followers", "median"),
        news_90d=("news_mentions_90d", "mean"),
    ).round(2).reindex(["D1", "D2", "D3", "NAIA"])
    print(agg)

    show = ["player_id", "name", "school", "division", "position", "height_in", "minutes_pg",
            "points_pg", "assists_pg", "turnovers_pg", "fg_pct", "three_pct",
            "recruiting_stars", "social_followers", "news_mentions_90d"]
    merged = planted.merge(players, on="player_id")
    for kind, title in (("gem", "Planted hidden gems (15)"), ("decoy", "Planted volume decoys (10)")):
        print(f"\n=== {title} ===")
        print(merged.loc[merged["kind"] == kind, show].sort_values(["division", "position"]).to_string(index=False))


def main() -> None:
    """Generate CSVs, load them through SeedDataSource into DuckDB, print a summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    players, planted = generate_players()
    players.to_csv(SEED_PLAYERS_CSV, index=False)
    planted.to_csv(SEED_PLANTED_CSV, index=False)

    source = SeedDataSource(SEED_PLAYERS_CSV)
    with get_connection() as con:
        n = load_players(con, source.fetch_players(), planted)
    print(f"Loaded {n} players into DuckDB ({len(planted)} planted rows in planted_gems).")
    print_summary(players, planted)


if __name__ == "__main__":
    main()
