"""ScoutAgent Streamlit UI. Talks to the FastAPI service over HTTP.

Run (with the API already running):  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `scout` imports work

import altair as alt  # noqa: E402  (bundled with Streamlit)
import httpx  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scout.config import API_URL  # noqa: E402

# (button label, full query). Labels stay short so the buttons don't truncate.
EXAMPLES = [
    ("D2 PG, passer, under 6'3\"", "Find a D2 point guard who'd translate to D1, strong passer, under 6'3\""),
    ("Shooting wings, small schools", "Best shooting wings from small schools"),
    ("NAIA/D3 rim protector", "NAIA or D3 big who protects the rim"),
]

st.set_page_config(page_title="ScoutAgent", page_icon="🏀", layout="wide")


# --------------------------------------------------------------------------- API client
class APIError(RuntimeError):
    pass


def api(method: str, path: str, timeout: float = 15.0, **kwargs: Any) -> dict[str, Any]:
    """Call the API; raise APIError with a readable message on any failure."""
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except httpx.ConnectError as e:
        port = urlparse(API_URL).port or 8000
        raise APIError(f"Can't reach the API at {API_URL}. Start it with: "
                       f"uvicorn api.main:app --port {port}") from e
    except httpx.TimeoutException as e:
        raise APIError(f"The API took longer than {timeout:.0f}s to respond") from e
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise APIError(f"API error {r.status_code}: {detail}")
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_player(player_id: str) -> dict[str, Any]:
    return api("GET", f"/players/{player_id}")


# --------------------------------------------------------------------------- state
st.session_state.setdefault("query", EXAMPLES[0][1])
st.session_state.setdefault("search", None)
st.session_state.setdefault("reports", {})


def run_search() -> None:
    try:
        st.session_state.search = api("POST", "/search", timeout=60, json={
            "query": st.session_state.query, "limit": 25,
            "include_gated": st.session_state.get("include_gated", False)})
        st.session_state.search_error = None
    except APIError as e:
        st.session_state.search, st.session_state.search_error = None, str(e)


def use_example(q: str) -> None:
    st.session_state.query = q
    run_search()


# --------------------------------------------------------------------------- header + search
st.title("🏀 ScoutAgent")
st.caption("Recruiting equity: find D1-caliber players at small schools that attention has missed.")

with st.form("search_form", border=False):
    c1, c2 = st.columns([6, 1])
    c1.text_input("Describe the player you need", key="query", label_visibility="collapsed",
                  placeholder="e.g. D2 point guard under 6'3\" who can pass")
    c2.form_submit_button("Search", type="primary", use_container_width=True, on_click=run_search)

ex_cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(ex_cols, EXAMPLES):
    col.button(f"Try: {label}", on_click=use_example, args=(q,), use_container_width=True, help=q)
st.toggle("Include players below the Translation gate", key="include_gated", on_change=run_search,
          help="By default, Underrated rankings only include players with Translation ≥ 50, "
               "so nobody ranks high just for being unknown.")

if st.session_state.get("search_error"):
    st.error(st.session_state.search_error)

res = st.session_state.search
if res is None:
    st.info("Run a search, or click an example above.")
    st.stop()

# --------------------------------------------------------------------------- how I understood it
parsed = res["parsed"]
chips = "".join(
    f"<span style='display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:12px;"
    f"background:rgba(49,104,214,.12);border:1px solid rgba(49,104,214,.35);font-size:.85rem'>{c}</span>"
    for c in parsed["chips"])
st.markdown(f"**How I understood your search:** {chips}", unsafe_allow_html=True)
by = {"llm": "Claude", "llm_retry": "Claude (after one retry)", "keyword": "keyword fallback"}[parsed["method"]]
note = f" · {'; '.join(parsed['notes'])}" if parsed["notes"] else ""
hidden = (f" · {res['excluded_by_gate']} more hidden (Translation < {res['gate']:.0f})"
          if res.get("gate") is not None and res["excluded_by_gate"] else "")
st.caption(f"Parsed by {by}{note} · {res['total_matches']} players match the filters{hidden}")

if not res["results"]:
    st.warning("No players match. Try loosening a filter or include gated players.")
    st.stop()

# --------------------------------------------------------------------------- results table
table = pd.DataFrame([{
    "Name": r["name"], "School": r["school"], "Div": r["division"], "Pos": r["position"],
    "Ht": r["height"], "Translation": r["translation_score"], "Underrated": r["underrated_score"],
    "Top factor": r["top_factor"],
} for r in res["results"]])
st.subheader("Shortlist")
event = st.dataframe(
    table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
    column_config={
        "Translation": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "Underrated": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "Top factor": st.column_config.TextColumn(width="large"),
    },
)
rows = event.selection.rows
if not rows:
    st.caption("Select a row to see the player profile and scouting report.")
    st.stop()

# --------------------------------------------------------------------------- player detail
pid = res["results"][rows[0]]["player_id"]
try:
    p = fetch_player(pid)
except APIError as e:
    st.error(str(e))
    st.stop()
s, raw, der, exp = p["summary"], p["raw"], p["derived"], p["explanation"]

st.divider()
st.subheader(f"{s['name']} · {s['height']} {s['position']} · {s['school']} ({s['division']}, {s['conference']})")

m = st.columns(8)
m[0].metric("PPG", f"{raw['points_pg']:.1f}")
m[1].metric("RPG", f"{raw['rebounds_pg']:.1f}")
m[2].metric("APG", f"{raw['assists_pg']:.1f}")
m[3].metric("TS%", f"{100 * der['ts_pct']:.1f}%")
m[4].metric("AST/TO", f"{der['ast_to']:.2f}")
m[5].metric("3P%", f"{100 * raw['three_pct']:.1f}%", help=f"{raw['three_att_pg']:.1f} attempts per game")
m[6].metric("Translation", f"{s['translation_score']:.1f}")
m[7].metric("Underrated", f"{s['underrated_score']:.1f}",
            help=f"Translation {s['translation_score']:.1f} − Attention {s['attention_score']:.1f}, rescaled to 0–100")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Percentile vs D1 players at his position** (after level adjustment)")
    chart_df = pd.DataFrame(p["percentiles_vs_d1_position"])
    chart_df["in score"] = chart_df["weight"].map(lambda w: "Counts toward Translation" if w > 0 else "Context only")
    bars = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="D1 position percentile"),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("in score:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["label", "percentile", "weight"],
    )
    avg = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(bars + avg, use_container_width=True)
    st.caption("Dashed line = D1 position average (50th percentile).")
with right:
    st.markdown("**Why this ranking**")
    for f in exp["top_factors"]:
        st.markdown(f"✅ {f['text']}")
    st.markdown(f"⚠️ {exp['top_concern']['text']}")
    for n in exp["notes"]:
        st.caption(n)
    st.caption(f"Attention inputs: {raw['recruiting_stars']}★ · {raw['social_followers']:,} followers · "
               f"{raw['news_mentions_90d']} news mentions (90 days) → Attention {s['attention_score']:.1f}")

# --------------------------------------------------------------------------- report
st.divider()
c1, c2 = st.columns([1, 1])
gen = c1.button("📝 Generate scouting report", type="primary")
regen = c2.button("Regenerate (skip cache)")
if gen or regen:
    with st.spinner("Writing report…"):
        try:
            st.session_state.reports[pid] = api("POST", f"/players/{pid}/report", timeout=120,
                                                params={"force": regen})
        except APIError as e:
            st.error(str(e))

rep = st.session_state.reports.get(pid)
if rep:
    src = f"Claude ({rep['model']})" if rep["method"] == "llm" else "offline template"
    st.caption(f"Written by {src}{' · from cache' if rep['cached'] else ''}"
               + (f" · {'; '.join(rep['notes'])}" if rep["notes"] else ""))
    if rep["flagged_numbers"]:
        st.warning("Fact check: these numbers are not in the player data: " + ", ".join(rep["flagged_numbers"]))
    else:
        st.success("Fact check passed: every number in this report appears in the player data.")
    if rep.get("unsupported_claims"):
        st.warning("Unsupported comparison (the report only has this player's data): "
                   + " · ".join(f"“{c}”" for c in rep["unsupported_claims"]))
    if rep["missing_sections"]:
        st.warning("Missing sections: " + ", ".join(rep["missing_sections"]))
    with st.container(border=True):
        st.markdown(rep["markdown"].replace("\n## ", "\n#### ").replace("## ", "#### ", 1))  # smaller headers
