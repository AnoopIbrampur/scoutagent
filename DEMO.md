# ScoutAgent: 3-minute demo script

Before you present, give yourself five minutes:

1. Start the API and the UI. Run one search and check that the line under the search bar says "understood by Claude". If it says "the keyword parser", there's no key set.
2. Run `python -m scripts.warm_cache 25` so the reports you're likely to open are already cached.
3. Skim the backup table at the bottom of this page.
4. Reload the page right before you start. The example pills rotate after every search, and a fresh page brings back the three this script uses.

Say up front that the data is synthetic. The players, schools and stats are generated, with a few hidden gems planted so we can check the scoring finds them. Nobody should walk away thinking these are real recruits.

---

## 0:00 to 0:15. The problem

> "Every year D1 programs miss players who could help them, because recruiting attention follows big programs, not performance. A D2 point guard with elite assist-to-turnover numbers might have zero stars and zero news coverage. ScoutAgent finds those players and shows its work."

## 0:15 to 1:15. Live search (60s)

1. Click the "D2 point guard, passer, under 6'3"" pill. Typing it yourself looks more live if you're comfortable with that.
2. Point at the "Searching for" pills. "This is how it read my request: D2, point guard, 6'2" or shorter, passing. Notice that 'translate to D1' didn't turn into a D1 filter."
3. Point at the line below. "25 players match, and 13 are hidden because their Translation Score is under 50. I don't want anyone ranking high just for being unknown." The Options menu can show them.
4. Point at the shortlist. "Underrated is how much better he is than the attention he gets. Translation is whether his production would hold up at D1."
5. Optional, if someone asks what else it understands: click More ideas. The pills swap to other searches, like a messy typed request, a height range, or a specific school.

## 1:15 to 2:00. The scouting report (45s)

1. Darius Ellison (D2 PG, 6'1") is already open on the right as the top match.
2. Under "Why this ranking": "A 4.62 assist-to-turnover ratio. Even after discounting for D2 competition, that's 99th percentile among D1 point guards."
3. Point at the chart. "Each bar is a percentile against D1 players at his position, after the level adjustment. The dashed line is the D1 average." If there's time, open the Synergy tab to show points per possession and his play types.
4. Switch to Scouting report and click Write scouting report. Scroll through it: snapshot, strengths, weaknesses, D1 projection, archetype, recommendation.
5. Point at the green Fact-checked pill. "Every number in this report is checked against his data. If the model made up a stat, we'd catch it and ask for a rewrite."

## 2:00 to 2:30. How the scoring works (30s)

> "Three steps. First, we adjust for level. A D2 season counts for about 80% of a D1 season, scaled by how strong the conference is. Second, we turn those adjusted stats into percentiles against D1 players at the same position, weighted by what matters there: decision-making for guards, rebounding and rim protection for bigs. Third, we subtract an attention score built from stars, followers and news. To test it, we planted 15 hidden gems in the data. All 15 land in the top 50, and it was a fair test because the scoring never sees which players were planted."

If someone pushes on that last point: the planted gems were built with an efficiency edge and the weights reward efficiency, so passing was always likely. It shows the pipeline works. It doesn't prove the weights are right, and say so before they do.

## 2:30 to 3:00. Impact and what's next (30s)

> "For a small-school player, getting found can decide whether he gets a scholarship. For a staff, this turns weeks of film triage into a shortlist. Next is real data. The ingestion layer already reads Synergy exports, so the next step is running it on a real export instead of generated players."

---

## If something breaks live

| What happened | What you'll see | What to do |
|---|---|---|
| Claude is down or slow | "understood by the keyword parser" under the search bar | Keep going. Say "the model isn't responding, so it's using a simpler parser." The three example queries give the same filters either way. A messy custom query might not. |
| The report doesn't generate | "Offline template" next to the report | It has the same sections and passes the same fact-check. Or open a player from the cached top 25. |
| No Wi-Fi at all | The app is local, only Claude needs a network | Leave the API key unset and restart the API. Everything runs offline. |
| The API crashed | "Can't reach the API…" under the search bar | Restart it with `uvicorn api.main:app --port 8000`. It's back in about two seconds. |
| The laptop or the UI dies | n/a | Play a screen recording of the full flow (make one the night before). Or fall back to the terminal with `python -m scripts.try_queries` and `python -m scripts.evaluate_scoring`. |

Don't demo a query you haven't run that day. Stick to the three example pills plus one custom query you've rehearsed.
