"""ScoutAgent Streamlit UI. Talks to the FastAPI service over HTTP.

Run (with the API already running):  streamlit run ui/app.py

Layout follows Apple's design principles (styles live in ui/style.css):
  - One clear path: describe a player -> scan the shortlist -> read the player.
  - Shortlist and player sit side by side, so detail appears next to what you picked.
  - The top match opens automatically; advanced controls live one level deeper
    (Options menu, Synergy section, report).
  - Status is shown as small capsules colored by meaning, not full-width alerts.
"""

from __future__ import annotations

import html
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
from scout.synergy import PERCENT_KEYS, RANK_KEYS, SYNERGY_COLUMNS  # noqa: E402

# Short pill label -> full query (labels stay short so pills don't wrap).
EXAMPLES = {
    "D2 point guard, passer, under 6'3\"": "Find a D2 point guard who'd translate to D1, strong passer, under 6'3\"",
    "Shooting wings, small schools": "Best shooting wings from small schools",
    "NAIA or D3 rim protector": "NAIA or D3 big who protects the rim",
}
ACCENT, NEUTRAL = "#0A84FF", "#8E8E93"  # system blue / system gray (work in light and dark)

st.set_page_config(page_title="ScoutAgent", page_icon="🏀", layout="wide")
_UI = Path(__file__).parent
_dark = (_UI / "style-dark.css").read_text()
_theme = getattr(st.context.theme, "type", None)  # Streamlit's active theme; None before the browser reports it
_dark_css = _dark if _theme == "dark" else "" if _theme == "light" else f"@media (prefers-color-scheme: dark) {{ {_dark} }}"
st.markdown(f"<style>{(_UI / 'style.css').read_text()}{_dark_css}</style>", unsafe_allow_html=True)


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


@st.cache_data(ttl=30, show_spinner=False)
def dataset_name() -> str | None:
    try:
        return api("GET", "/health", timeout=5).get("dataset")
    except APIError:
        return None


# --------------------------------------------------------------------------- small view helpers
def chips(items: list[tuple[str, str]], label: str | None = None) -> None:
    """Row of status capsules. tone: accent | good | warn | bad | '' (neutral)."""
    parts = [f"<span class='sa-label'>{html.escape(label)}</span>"] if label else []
    for text, tone in items:
        dot = "<span class='dot'></span>" if tone in ("good", "warn", "bad") else ""
        parts.append(f"<span class='sa-chip {tone}'>{dot}{html.escape(text)}</span>")
    st.markdown(f"<div class='sa-chips'>{''.join(parts)}</div>", unsafe_allow_html=True)


def reason_rows(factors: list[str], concern: str) -> None:
    """'Why this ranking' as a grouped list: + strengths, then the main concern."""
    def row(icon: str, cls: str, text: str) -> str:
        what, _, detail = text.partition(": ")
        return (f"<div class='sa-row'><div class='icon {cls}'>{icon}</div><div>"
                f"<div class='what'>{html.escape(what)}</div>"
                f"<div class='detail'>{html.escape(detail)}</div></div></div>")
    body = "".join(row("+", "plus", f) for f in factors) + row("−", "minus", concern)
    st.markdown(f"<div class='sa-list'>{body}</div>", unsafe_allow_html=True)


def fmt_score(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def synergy_frame(rows: list[dict[str, Any]], first_col: str = "STAT") -> pd.DataFrame:
    """Rows -> DataFrame with Synergy's column labels and display formatting."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for key in PERCENT_KEYS:
        df[key] = df[key].map(lambda v: "–" if pd.isna(v) else f"{100 * v:.1f}%")
    for key in RANK_KEYS:
        df[key] = df[key].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}%")
    df["ppp"] = df["ppp"].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}")
    df["ssm"] = df["ssm"].map(lambda v: "–" if pd.isna(v) else f"{v:+.2f}")
    for key in ("ssq", "pps", "fta_per_fga"):
        df[key] = df[key].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}")
    df["rating"] = df["rating"].fillna("–")
    labels = {"situation": first_col} | dict(SYNERGY_COLUMNS)
    return df[list(labels)].rename(columns=labels)


def percentile_chart(points: list[dict[str, Any]]) -> alt.LayerChart:
    """Horizontal bars vs a dashed D1-average line. Lighter bars are context only."""
    df = pd.DataFrame(points)
    df["counts"] = df["weight"] > 0
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=240, domain=False, ticks=False, labelPadding=8)),
        x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title=None,
                axis=alt.Axis(values=[0, 50, 100], grid=False, domain=False, ticks=False)),
        tooltip=[alt.Tooltip("label:N", title="Stat"), alt.Tooltip("percentile:Q", title="Percentile")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        color=alt.condition("datum.counts", alt.value(ACCENT), alt.value(NEUTRAL)),
        opacity=alt.condition("datum.counts", alt.value(1.0), alt.value(0.45)),
    )
    avg = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[3, 3], opacity=0.5).encode(x="x:Q")
    return ((bars + avg).properties(height=max(160, 26 * len(df)), background="transparent")
            .configure_view(stroke=None))


# --------------------------------------------------------------------------- state
st.session_state.setdefault("query", "")
st.session_state.setdefault("search", None)
st.session_state.setdefault("reports", {})


def run_search() -> None:
    if not st.session_state.query.strip():
        return
    try:
        st.session_state.search = api("POST", "/search", timeout=60, json={
            "query": st.session_state.query, "limit": 25,
            "include_gated": st.session_state.get("include_gated", False)})
        st.session_state.search_error = None
    except APIError as e:
        st.session_state.search, st.session_state.search_error = None, str(e)


def pick_example() -> None:
    label = st.session_state.example
    if label:
        st.session_state.query = EXAMPLES[label]
        st.session_state.example = None  # behave like a button, not a toggle
        run_search()


# --------------------------------------------------------------------------- hero
st.markdown(
    "<p class='sa-overline'>ScoutAgent</p>"
    "<h1 class='sa-display'>Find the players attention missed.</h1>"
    "<p class='sa-lede'>Describe who you need. ScoutAgent ranks small-school players by how well their "
    "production should translate to D1, and how little anyone is watching.</p>",
    unsafe_allow_html=True,
)
if dataset_name() == "real":
    chips([("Real players from Synergy exports", "accent"),
           ("Compared against a synthetic D1 baseline", "warn")])

# --------------------------------------------------------------------------- floating search bar
with st.container(key="searchbar"):
    with st.form("search_form", border=False):
        c1, c2 = st.columns([7, 1], vertical_alignment="bottom")
        c1.text_input("Describe the player you need", key="query", label_visibility="collapsed",
                      placeholder="D2 point guard under 6'3\" who can really pass")
        c2.form_submit_button("Search", type="primary", width="stretch", on_click=run_search)
    p1, p2 = st.columns([7, 1], vertical_alignment="center")
    p1.pills("Try", list(EXAMPLES), key="example", on_change=pick_example, label_visibility="collapsed")
    with p2.popover("Options", width="stretch"):
        st.toggle("Include players below the Translation gate", key="include_gated", on_change=run_search,
                  help="By default, Underrated rankings only include players with Translation ≥ 50, "
                       "so nobody ranks high just for being unknown.")

if st.session_state.get("search_error"):
    st.error(st.session_state.search_error, icon=":material/wifi_off:")

res = st.session_state.search
if res is None:
    with st.container(key="empty"):
        st.markdown("<p class='sa-section'>Start with a description</p>", unsafe_allow_html=True)
        st.caption("Try a level, a position, a size, and what he should be good at, like "
                   "“NAIA wing who shoots it, at least 6'5”.” Or pick one of the examples above.")
    st.stop()

# --------------------------------------------------------------------------- how I understood it
parsed = res["parsed"]
chips([(c, "accent" if not c.startswith("Sorted") else "") for c in parsed["chips"]], label="Searching for")
by = {"llm": "Claude", "llm_retry": "Claude (after one retry)", "keyword": "the keyword parser"}[parsed["method"]]
meta = f"{res['total_matches']} players match · understood by {by}"
if res.get("gate") is not None and res["excluded_by_gate"]:
    meta += f" · {res['excluded_by_gate']} below Translation {res['gate']:.0f} hidden (see Options)"
st.caption(meta + (f" · {'; '.join(parsed['notes'])}" if parsed["notes"] else ""))

if not res["results"]:
    with st.container(key="empty"):
        st.markdown("<p class='sa-section'>No players match</p>", unsafe_allow_html=True)
        st.caption("Loosen one filter, or turn on “Include players below the Translation gate” in Options.")
    st.stop()

# --------------------------------------------------------------------------- shortlist | player
left, right = st.columns([11, 13], gap="large")

with left:
    st.markdown(f"<p class='sa-section'>Shortlist <span class='sa-label'>· {len(res['results'])}</span></p>",
                unsafe_allow_html=True)
    # Most important first: the scores sit next to the name, so they stay visible in a narrow list.
    table = pd.DataFrame([{
        "Name": r["name"], "Underrated": r["underrated_score"], "Translation": r["translation_score"],
        "Level": f"{r['division']} · {r['position']}", "Ht": r["height"], "School": r["school"],
    } for r in res["results"]])
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 640),
        on_select="rerun", selection_mode="single-row", key=f"shortlist-{hash(st.session_state.query)}",
        column_config={  # fixed widths so the scores never scroll out of view
            "Name": st.column_config.TextColumn(width=128),
            "School": st.column_config.TextColumn(width=132),
            "Level": st.column_config.TextColumn(width=66),
            "Ht": st.column_config.TextColumn(width=50),
            "Translation": st.column_config.NumberColumn("Transl.", format="%.1f", width=62,
                                                         help="Translation: would his production hold up at D1? 0–100"),
            "Underrated": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f", width=118,
                                                          help="Translation minus attention, rescaled 0–100"),
        },
    )
    st.caption("Select a row to open that player. The top match is open by default.")

# The common path first: with nothing selected, show the top match.
picked = event.selection.rows
idx = picked[0] if picked else 0
pid = res["results"][idx]["player_id"]

with right:
    try:
        p = fetch_player(pid)
    except APIError as e:
        st.error(str(e))
        st.stop()
    s, raw, der, exp = p["summary"], p["raw"], p["derived"], p["explanation"]

    with st.container(key="detail"):
        tag = "Top match" if not picked else f"#{idx + 1} on the shortlist"
        st.markdown(
            f"<p class='sa-overline'>{tag}</p><h2 class='sa-title'>{html.escape(s['name'])}</h2>"
            f"<p class='sa-meta'>{html.escape(s['height'])} {s['position']} · {html.escape(s['school'])} · "
            f"{s['division']} {html.escape(s['conference'] or '')} · {s['class_year']}</p>",
            unsafe_allow_html=True,
        )

        m = st.columns(4)
        m[0].metric("Translation", f"{s['translation_score']:.1f}", help="Would his production hold up at D1? 0–100")
        m[1].metric("Underrated", fmt_score(s["underrated_score"]),
                    help=("No attention data for this player" if s["underrated_score"] is None else
                          f"Translation {s['translation_score']:.1f} − Attention {s['attention_score']:.1f}, rescaled 0–100"))
        m[2].metric("PPG", f"{raw['points_pg']:.1f}")
        m[3].metric("TS%", f"{100 * der['ts_pct']:.1f}%")

        sections = ["Overview", "Synergy", "Scouting report"] if p.get("synergy") else ["Overview", "Scouting report"]
        view = st.segmented_control("Section", sections, default="Overview", key=f"view-{pid}",
                                    label_visibility="collapsed") or "Overview"

        # ------------------------------------------------------------ overview
        if view == "Overview":
            st.markdown("<p class='sa-section'>Why this ranking</p>", unsafe_allow_html=True)
            reason_rows([f["text"] for f in exp["top_factors"]], exp["top_concern"]["text"])

            st.markdown("<p class='sa-section' style='margin-top:18px'>Against D1 players at his position</p>",
                        unsafe_allow_html=True)
            st.altair_chart(percentile_chart(p["percentiles_vs_d1_position"]), width="stretch")
            st.caption("Percentile after the level adjustment; dashed line is the D1 average. "
                       "Blue bars count toward Translation; gray bars are context.")

            box = st.columns(4)
            box[0].metric("RPG", f"{raw['rebounds_pg']:.1f}")
            box[1].metric("APG", f"{raw['assists_pg']:.1f}")
            box[2].metric("AST/TO", f"{der['ast_to']:.2f}")
            box[3].metric("3P%", f"{100 * raw['three_pct']:.1f}%" if raw["three_pct"] == raw["three_pct"] else "–",
                          help=f"{raw['three_att_pg']:.1f} attempts per game")

            notes = list(exp["notes"])
            if s["attention_score"] is None:
                notes.append("Attention: none loaded (blank means unknown, not zero)")
            else:
                f = lambda v: "?" if v is None else f"{v:,.0f}"  # noqa: E731
                notes.append(f"Attention {s['attention_score']:.1f} from {f(raw['recruiting_stars'])}★, "
                             f"{f(raw['social_followers'])} followers, {f(raw['news_mentions_90d'])} news mentions (90 days)")
            st.caption(" · ".join(notes))

        # ------------------------------------------------------------ synergy
        elif view == "Synergy":
            syn = p["synergy"]
            o = next((r for r in syn.get("offense", []) if r["situation"] == "All Possessions"), None)
            d = next((r for r in syn.get("defense", []) if r["situation"] == "All Possessions"), None)
            sm = st.columns(4)
            if o:
                sm[0].metric("Points / poss.", f"{o['ppp']:.3f}", help=f"{o['rating'] or 'unrated'} · rank {o['ppp_rank']}")
                sm[1].metric("Shot-making", f"{o['ssm']:+.2f}",
                             help="SSM: points per shot above what his shot quality (SSQ) predicts")
                sm[2].metric("Turnover rate", f"{100 * o['to_pct']:.1f}%")
            if d:
                sm[3].metric("Allowed / poss.", f"{d['ppp']:.3f}", help=f"{d['rating'] or 'unrated'} · rank {d['ppp_rank']}")
            ratings = [(f"Offense: {o['rating']}", "good" if o["rating"] in ("Excellent", "Very Good") else "")] if o and o["rating"] else []
            if d and d["rating"]:
                ratings.append((f"Defense: {d['rating']}", "good" if d["rating"] in ("Excellent", "Very Good") else
                                "warn" if d["rating"] in ("Below Average", "Poor") else ""))
            if ratings:
                chips(ratings)

            options = {"Offense": "offense", "Defense": "defense",
                       "Play types": "playtypes_offense", "Defensive play types": "playtypes_defense"}
            options = {k: v for k, v in options.items() if syn.get(v)}
            which = st.segmented_control("Table", list(options), default="Offense", key=f"syn-{pid}",
                                         label_visibility="collapsed") or "Offense"
            key = options[which]
            st.dataframe(synergy_frame(syn[key], first_col="PLAY TYPE" if key.startswith("playtypes") else "STAT"),
                         hide_index=True, width="stretch")
            st.caption("Same layout as a Synergy Player Details export. Ranks are percentiles across all players; "
                       "“–” means fewer than 10 possessions."
                       + (" Play-type layout is assumed until a real play-type export confirms it."
                          if key.startswith("playtypes") else ""))

        # ------------------------------------------------------------ report
        else:
            rep = st.session_state.reports.get(pid)
            b1, b2 = st.columns([2, 1])
            write = b1.button("Write scouting report" if not rep else "Show cached report", type="primary",
                              width="stretch", icon=":material/description:")
            redo = b2.button("Regenerate", width="stretch", disabled=rep is None,
                             help="Skip the cache and write a fresh report")
            if write or redo:
                with st.spinner("Writing report…"):
                    try:
                        rep = st.session_state.reports[pid] = api(
                            "POST", f"/players/{pid}/report", timeout=120, params={"force": redo})
                    except APIError as e:
                        st.error(str(e))
            if rep:
                status = [("Fact-checked" if not rep["flagged_numbers"] else
                           f"Unverified numbers: {', '.join(rep['flagged_numbers'])}",
                           "good" if not rep["flagged_numbers"] else "bad")]
                if rep.get("unsupported_claims"):
                    status.append((f"{len(rep['unsupported_claims'])} unsupported comparison(s)", "warn"))
                if rep["missing_sections"]:
                    status.append((f"Missing: {', '.join(rep['missing_sections'])}", "warn"))
                status.append((f"{'Claude' if rep['method'] == 'llm' else 'Offline template'}"
                               f"{' · cached' if rep['cached'] else ''}", ""))
                chips(status)
                for claim in rep.get("unsupported_claims", []):
                    st.caption(f"Unsupported: “{claim}”")
                with st.container(key="report"):
                    st.markdown(rep["markdown"].replace("\n## ", "\n#### ").replace("## ", "#### ", 1))
                if rep["notes"]:
                    st.caption(" · ".join(rep["notes"]))
            else:
                st.caption("Six sections: snapshot, strengths, weaknesses, D1 projection, archetype, "
                           "and a Pursue / Monitor / Pass call. Every number is checked against his data.")
