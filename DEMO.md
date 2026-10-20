# ScoutAgent: 3-minute demo script

**Before you present (5 minutes):**
1. Start the API and UI. Open the UI, run one search, and check that the caption says "Parsed by Claude".
2. Run `python -m scripts.warm_cache 25`, so reports for the likely clicks come from cache instantly.
3. Have the backups ready (bottom of this page).

---

## 0:00–0:15 · The problem hook

> "Every year D1 programs miss players who could help them, because recruiting attention
> follows big programs, not performance. A D2 point guard with elite assist-to-turnover numbers
> might have zero stars and zero news coverage. ScoutAgent finds those players and explains why."

## 0:15–1:15 · Live search (60s)

1. Click the **D2 point guard, passer, under 6'3"** pill (or type it, which looks more live).
2. Point at the "Searching for" capsules: *"Here's how it understood me: Division D2, point guard,
   6'2" or shorter, emphasis on passing. Notice 'translate to D1' did **not** become a D1 filter."*
3. Point at the line under it: *"25 players match. 13 are hidden because their Translation Score is
   below 50. We don't let anyone rank high just for being unknown."* (Options can show them.)
4. Point at the shortlist: *"Underrated is how much better he is than the attention he gets.
   Translation asks whether his production would hold up at D1."*

## 1:15–2:00 · Scouting report (45s)

1. **Darius Ellison** (D2 PG, 6'1") is already open on the right as the top match.
2. "Why this ranking": *"A 4.62 assist-to-turnover ratio. After we discount for D2 competition,
   that's still 99th percentile among D1 point guards."*
3. Chart: *"Every bar is a percentile against D1 players at his position, after the level
   adjustment. The dashed line is D1 average."* (Optional: switch to **Synergy** to show
   points per possession and his play types.)
4. Switch to **Scouting report** and click **Write scouting report**. Scroll through it: snapshot,
   strengths, weaknesses, D1 projection, archetype, recommendation.
5. Point at the green **Fact-checked** pill: *"Every number in this report is checked against the
   source data. If the model invents a stat, we catch it and regenerate."*

## 2:00–2:30 · How the scoring works (30s)

> "Three steps, all transparent. First, we adjust production for level: a D2 season counts for about
> 80% of a D1 one, scaled by conference strength. Second, we turn the adjusted stats into percentiles
> against real D1 players at the same position, weighted by what matters for the position: decision-making
> for point guards, rebounding and rim protection for bigs. Third, we subtract an attention score
> built from stars, followers and news. We tested it with planted hidden gems: all 15 rank in our
> top 50."

## 2:30–3:00 · Impact and what's next (30s)

> "For a small-school player, being found is the difference between a scholarship and not. For
> a staff, this is a shortlist in seconds instead of weeks of film triage. Next: real stats
> through a licensed data source (the ingestion layer is already pluggable), multi-season trends,
> and agents that watch for breakout performances and alert staffs."

---

## Backup plan if things fail live

| Failure | What you see | What to do |
|---|---|---|
| Claude API down or slow | Caption: "Parsed by keyword fallback · LLM unavailable…" | Keep going. Say: *"The model is unavailable, so it falls back to a deterministic parser. Same results."* The example queries are tuned for the fallback. |
| Report generation fails | Caption: "Written by offline template" | Same structure, guaranteed fact-check pass. Or click a player warmed in the cache (top 25). |
| Wi-Fi dead entirely | UI and API run locally; only Claude needs the network | Unset the key (`ANTHROPIC_API_KEY=`) and restart the API. Everything runs offline. |
| API process crashed | Error under the search bar: "Can't reach the API…" | Restart: `uvicorn api.main:app --port 8000`. It loads in about 2s. |
| Laptop or UI fails | n/a | Screen recording of the full flow (record it the night before) plus terminal fallback: `python -m scripts.try_queries` and `python -m scripts.evaluate_scoring`. |

**Never** demo a query you haven't tried that day. Stick to the three example buttons, plus one
custom query you've rehearsed.
