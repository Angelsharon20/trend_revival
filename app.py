"""
app.py - Trend Revival analytical dashboard (Streamlit + Plotly, custom dark SaaS theme).

Run:  streamlit run app.py
First launch builds the whole pipeline automatically (MongoDB if reachable,
otherwise the pure-Python MapReduce fallback).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import comeback_detector
import config
import dataset_generator
import mapreduce_engine
import preprocess
from comeback_detector import Params, detect_all

C = config.COLORS
NAV = ["Overview Dashboard", "Trend Timeline Explorer", "Comeback Detector Analytics",
       "Topic Comparison Matrix", "Data & Methodology Blueprint"]
NAV_ICONS = {NAV[0]: "◈", NAV[1]: "◔", NAV[2]: "◬", NAV[3]: "▦", NAV[4]: "❖"}
EARLY, RES = config.DETECTOR["early_months"], config.DETECTOR["resurgence_months"]
DATES = [f"{m}-01" for m in config.MONTHS]

st.set_page_config(page_title="Trend Revival | Comeback Analytics", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")


# =========================================================================== #
# Styling
# =========================================================================== #
def inject_css() -> None:
    root = (f":root{{--bg:{C['bg']};--card:{C['card']};--card2:{C['card_alt']};--line:{C['line']};"
            f"--text:{C['text']};--muted:{C['muted']};--cyan:{C['cyan']};--purple:{C['purple']};"
            f"--green:{C['green']};--pink:{C['pink']};--amber:{C['amber']};--red:{C['red']};}}")
    css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; color:var(--text); }
.stApp { background: radial-gradient(1100px 560px at 88% -12%, rgba(0,229,255,.08), transparent 60%),
         radial-gradient(900px 500px at -8% 8%, rgba(179,136,255,.07), transparent 55%), var(--bg); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1520px; }
h1,h2,h3,h4 { color:var(--text); letter-spacing:-.01em; }
hr { border-color:var(--line); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background:#090D13; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding-top:.6rem; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap:.35rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { background:transparent; border:1px solid transparent;
  border-radius:10px; padding:.62rem .85rem; margin:0; transition:all .18s ease; width:100%; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background:rgba(0,229,255,.06); border-color:rgba(0,229,255,.25); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background:linear-gradient(90deg, rgba(0,229,255,.17), rgba(179,136,255,.08));
  border-color:rgba(0,229,255,.55); box-shadow:0 0 18px rgba(0,229,255,.13); }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] p { font-size:.9rem; font-weight:500; }
.tr-brand { display:flex; align-items:center; gap:.7rem; padding:.2rem .3rem 1.1rem .3rem; }
.tr-logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:1.2rem;
  background:linear-gradient(135deg,var(--cyan),var(--purple)); color:#05080c; font-weight:800;
  box-shadow:0 0 22px rgba(0,229,255,.35); }
.tr-brand-name { font-weight:700; font-size:1.05rem; line-height:1.1; }
.tr-brand-sub { color:var(--muted); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; }
.tr-side-label { color:var(--muted); font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; margin:1.1rem 0 .4rem .3rem; }
.tr-status { border:1px solid var(--line); background:var(--card2); border-radius:12px; padding:.75rem .9rem; font-size:.78rem; color:var(--muted); }
.tr-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.5rem; box-shadow:0 0 10px currentColor; }

/* ---------- page header ---------- */
.tr-title { font-size:1.75rem; font-weight:700; margin:0; }
.tr-title span { background:linear-gradient(90deg,var(--cyan),var(--purple)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.tr-sub { color:var(--muted); font-size:.92rem; margin:.15rem 0 1.2rem 0; }

/* ---------- KPI cards ---------- */
.tr-kpi { background:linear-gradient(180deg,var(--card),var(--card2)); border:1px solid var(--line);
  border-top:2px solid var(--accent); border-radius:14px; padding:1rem 1.1rem .7rem 1.1rem;
  box-shadow:0 0 0 1px rgba(255,255,255,.01), 0 10px 30px rgba(0,0,0,.35), 0 0 22px -12px var(--accent);
  transition:transform .18s ease, box-shadow .18s ease; margin-bottom:.4rem; }
.tr-kpi:hover { transform:translateY(-2px); box-shadow:0 12px 34px rgba(0,0,0,.45), 0 0 26px -8px var(--accent); }
.tr-kpi-label { color:var(--muted); font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600; }
.tr-kpi-value { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.95rem; margin:.25rem 0 .1rem 0; }
.tr-kpi-delta { font-size:.76rem; font-weight:600; margin-bottom:.35rem; }
.up { color:var(--green); } .down { color:var(--red); } .flat { color:var(--muted); }
.tr-kpi svg { display:block; width:100%; height:46px; }

/* ---------- panels / charts ---------- */
div[data-testid="stPlotlyChart"] { background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:.4rem .5rem; box-shadow:0 10px 30px rgba(0,0,0,.30); }
div[class*="st-key-panel-feed"] { background:var(--card); border:1px solid var(--line) !important; border-radius:14px;
  box-shadow:0 10px 30px rgba(0,0,0,.30); padding:.4rem .6rem; }
.tr-panel-title { font-weight:600; font-size:1rem; margin:.1rem 0 .6rem .1rem; }
.tr-panel-title small { color:var(--muted); font-weight:400; margin-left:.4rem; }

/* ---------- feed ---------- */
.tr-feed { border:1px solid var(--line); border-left:3px solid var(--tone); background:var(--card2); border-radius:12px;
  padding:.7rem .85rem; margin:.5rem 0 .25rem 0; }
.tr-feed-tag { font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; color:var(--tone); }
.tr-feed-title { font-weight:600; font-size:.95rem; margin:.15rem 0 .2rem 0; }
.tr-feed-body { color:var(--muted); font-size:.8rem; line-height:1.4; }

/* ---------- table grid ---------- */
div[class*="st-key-thead"] { background:var(--card2); border:1px solid var(--line); border-radius:12px 12px 0 0; padding:.55rem .9rem; margin-top:.3rem; }
div[class*="st-key-row-"] { background:var(--card); border-left:1px solid var(--line); border-right:1px solid var(--line);
  border-bottom:1px solid #1d2531; padding:.35rem .9rem; }
div[class*="st-key-row-"]:hover { background:#1a212b; }
.tr-th { color:var(--muted); font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; }
.tr-td { font-size:.88rem; padding-top:.35rem; }
.tr-mono { font-family:'JetBrains Mono',monospace; }
.badge { display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.06em;
  border:1px solid currentColor; text-transform:uppercase; }
.badge-High { color:var(--pink); background:rgba(255,77,166,.10); box-shadow:0 0 12px rgba(255,77,166,.25); }
.badge-Medium { color:var(--amber); background:rgba(255,194,77,.10); }
.badge-Low { color:var(--cyan); background:rgba(0,229,255,.09); }
.badge-None { color:var(--muted); background:rgba(139,152,169,.08); }

/* ---------- stage cards / pipeline ---------- */
.tr-stage { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; height:100%; }
.tr-stage b { font-size:.95rem; } .tr-stage p { color:var(--muted); font-size:.8rem; margin:.25rem 0 0 0; }
.ok { color:var(--green); } .bad { color:var(--red); }
.tr-flow { display:flex; flex-wrap:wrap; gap:.6rem; margin:.4rem 0 1rem 0; }
.tr-node { flex:1 1 150px; background:var(--card); border:1px solid var(--line); border-top:2px solid var(--c); border-radius:12px; padding:.7rem .85rem; }
.tr-node small { color:var(--muted); display:block; font-size:.72rem; margin-top:.15rem; }
.tr-node .n { font-family:'JetBrains Mono',monospace; color:var(--c); font-size:.72rem; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button { background:rgba(0,229,255,.06); color:var(--cyan); border:1px solid rgba(0,229,255,.45);
  border-radius:9px; font-weight:600; font-size:.8rem; padding:.3rem .9rem; transition:all .16s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { background:rgba(0,229,255,.16); border-color:var(--cyan);
  color:#fff; box-shadow:0 0 16px rgba(0,229,255,.35); }
.stButton > button:focus:not(:active) { border-color:var(--cyan); color:var(--cyan); }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:var(--card2) !important; border-color:var(--line) !important; border-radius:10px !important; }
div[data-baseweb="tag"] { background:rgba(179,136,255,.18) !important; }
label, .stSlider label, .stMultiSelect label, .stSelectbox label, .stToggle label { color:var(--muted) !important; font-size:.78rem !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
code, pre { font-family:'JetBrains Mono',monospace !important; }
div[data-testid="stCode"] { border:1px solid var(--line); border-radius:12px; }
div[data-testid="stStatusWidget"], div[data-testid="stExpander"] { border-color:var(--line); }
"""
    st.markdown(f"<style>{root}{css}</style>", unsafe_allow_html=True)


# =========================================================================== #
# Small UI helpers
# =========================================================================== #
def sparkline(values, color: str, uid: str) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return ""
    w, h = 260, 46
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 5 - (v - lo) / span * (h - 12)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"M{xs[0]:.1f},{h} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f" L{xs[-1]:.1f},{h} Z"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{color}" stop-opacity=".35"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#g{uid})"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.2" fill="{color}"/></svg>')


def kpi_card(label: str, value: str, delta: str, delta_cls: str, spark, color: str, uid: str) -> str:
    return "".join([f'<div class="tr-kpi" style="--accent:{color}">', f'<div class="tr-kpi-label">{label}</div>',
                    f'<div class="tr-kpi-value">{value}</div>', f'<div class="tr-kpi-delta {delta_cls}">{delta}</div>',
                    sparkline(spark, color, uid), "</div>"])


def badge(priority: str) -> str:
    return f'<span class="badge badge-{priority}">{priority if priority != "None" else "Not a comeback"}</span>'


def page_header(title: str, subtitle: str) -> None:
    first, _, rest = title.partition(" ")
    st.markdown(f'<h1 class="tr-title"><span>{first}</span> {rest}</h1><div class="tr-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def period_label(period: str | None) -> str:
    return datetime.strptime(period, "%Y-%m").strftime("%b %Y") if period else "-"


def pct_delta(now: float, before: float) -> tuple[str, str]:
    if before <= 0:
        return "n/a", "flat"
    d = (now / before - 1) * 100
    return f"{'▲' if d >= 0 else '▼'} {abs(d):.1f}% vs prior 6 months", ("up" if d >= 0 else "down")


def goto_topic(topic: str) -> None:
    st.session_state["selected_topic"] = topic
    st.session_state["nav"] = NAV[1]


def style_fig(fig: go.Figure, title: str = "", height: int = 430, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15, color=C["text"])), height=height,
        paper_bgcolor=C["card"], plot_bgcolor=C["card"], font=dict(family="Inter", color=C["muted"], size=12),
        margin=dict(l=16, r=16, t=70 if title else 40, b=16), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["bg"], bordercolor=C["line"], font=dict(color=C["text"])))
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False, linecolor=C["line"], showspikes=True, spikecolor=C["muted"], spikethickness=1)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, linecolor=C["line"])
    return fig


def add_stage_overlays(fig: go.Figure) -> None:
    """Shade the 3 detector windows: Initial Peak / Decline-Quiet / Resurgence."""
    n = len(DATES)
    stages = [(0, EARLY, C["cyan"], "① INITIAL PEAK"), (EARLY, n - RES, C["purple"], "② DECLINE / QUIET PERIOD"),
              (n - RES, n - 1, C["green"], "③ RESURGENCE")]
    for a, b, color, label in stages:
        fig.add_vrect(x0=DATES[a], x1=DATES[b], fillcolor=color, opacity=0.06, line_width=0, layer="below")
        mid = pd.Timestamp(DATES[a]) + (pd.Timestamp(DATES[b]) - pd.Timestamp(DATES[a])) / 2
        fig.add_annotation(x=mid.strftime("%Y-%m-%d"), y=1.0, yref="paper", yanchor="bottom", text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=10, color=color))


# =========================================================================== #
# Pipeline bootstrap + data loading
# =========================================================================== #
def run_pipeline(log, use_mongo: bool) -> None:
    log("① Generating 124,862-post dataset …")
    dataset_generator.generate_dataset()
    if use_mongo:
        try:
            import ingest_data

            log("② Ingesting into MongoDB raw_posts …")
            ingest_data.ingest(reset=True, verbose=False)
            log("③ Cleaning posts + exploding topic_mentions (indexes) …")
            preprocess.run_mongo(workers=1, verbose=False)
            log("④ Running server-side MongoDB MapReduce …")
            mapreduce_engine.run(verbose=False)
            log("⑤ Detecting comebacks …")
            comeback_detector.run(verbose=False)
            return
        except Exception as exc:  # noqa: BLE001
            log(f"MongoDB path failed ({exc}); continuing with the local engine …")
    log("② Cleaning posts + exploding topic_mentions (local) …")
    preprocess.run_local(workers=1, verbose=False)
    log("③ Running Python Map → Shuffle → Reduce fallback …")
    mapreduce_engine.run(force_python=True, verbose=False)
    log("④ Detecting comebacks …")
    comeback_detector.run(verbose=False)


def ensure_pipeline() -> None:
    if mapreduce_engine.period_counts_available() and comeback_detector.results_available():
        return
    with st.status("Building the analytical pipeline (first launch only) …", expanded=True) as status:
        run_pipeline(st.write, use_mongo=config.get_db() is not None)
        status.update(label="Pipeline ready", state="complete", expanded=False)
    st.cache_data.clear()


@st.cache_data(show_spinner=False)
def load_data(version: int):
    counts = mapreduce_engine.load_period_counts()
    df = pd.DataFrame(counts, index=pd.Index(config.MONTHS, name="period"))
    results = comeback_detector.load_results()
    stats = preprocess.load_stats()
    return df, results, stats


@st.cache_data(show_spinner=False)
def live_detection(dormancy: float, recovery: float, quiet: int, _df: pd.DataFrame):
    params = Params(dormancy_ratio=dormancy, recovery_ratio=recovery, min_quiet_months=quiet)
    return detect_all({t: _df[t].tolist() for t in _df.columns}, params)


# =========================================================================== #
# Views
# =========================================================================== #
def main_chart(df: pd.DataFrame, rmap: dict, topics: list[str], normalize: bool) -> go.Figure:
    fig = go.Figure()
    for i, t in enumerate(topics):
        y = df[t].astype(float)
        peak = rmap.get(t, {}).get("historical_peak") or y.iloc[:EARLY].max() or 1
        yy = y / peak * 100 if normalize else y
        color = config.CHART_COLORS[i % len(config.CHART_COLORS)]
        fig.add_trace(go.Scatter(x=DATES, y=yy, name=config.display_name(t), mode="lines",
                                 line=dict(color=color, width=2.6, shape="spline", smoothing=0.5),
                                 hovertemplate="%{y:,.0f}" + ("%" if normalize else "") + "<extra>" + config.display_name(t) + "</extra>"))
    add_stage_overlays(fig)
    if normalize:
        fig.add_hline(y=25, line=dict(color=C["red"], dash="dot", width=1), annotation_text="25% dormancy line",
                      annotation_font=dict(color=C["red"], size=10), annotation_position="bottom left")
        fig.add_hline(y=50, line=dict(color=C["green"], dash="dot", width=1), annotation_text="50% recovery line",
                      annotation_font=dict(color=C["green"], size=10), annotation_position="top left")
    lead = next((t for t in topics if rmap.get(t, {}).get("is_comeback")), None)
    if lead:
        r = rmap[lead]
        yv = (100 if normalize else r["historical_peak"])
        fig.add_annotation(x=f"{r['peak_period']}-01", y=yv, text=f"Historical peak<br><b>{r['historical_peak']:,}</b>", ax=40, ay=-34,
                           showarrow=True, arrowcolor=C["cyan"], font=dict(color=C["cyan"], size=11), bgcolor=C["bg"], bordercolor=C["cyan"])
        ylast = df[lead].iloc[-3:].mean() / (r["historical_peak"] if normalize else 1) * (100 if normalize else 1)
        fig.add_annotation(x=DATES[-2], y=ylast, text=f"Resurgence<br><b>{r['spike_velocity']:.0%}</b> of peak", ax=-70, ay=-46,
                           showarrow=True, arrowcolor=C["green"], font=dict(color=C["green"], size=11), bgcolor=C["bg"], bordercolor=C["green"])
    return style_fig(fig, "Trend lifecycle: peak → quiet valley → resurgence", 470)


def build_feed(df: pd.DataFrame, rdf: pd.DataFrame, limit: int = 7) -> list[dict]:
    feed = []
    for _, r in rdf[rdf.is_comeback].head(limit).iterrows():
        tone = config.PRIORITY_COLORS[r.priority]
        feed.append(dict(topic=r.topic, tone=tone, tag=f"{r.priority} priority comeback", title=config.display_name(r.topic),
                         body=f"Back to {r.spike_velocity:.0%} of its {period_label(r.peak_period)} peak after {r.quiet_months} months below "
                              f"{config.DETECTOR['dormancy_ratio']:.0%}. Recent avg {r.recent_count:,.0f} posts/month."))
    last3, prev3 = df.iloc[-3:].mean(), df.iloc[-6:-3].mean()
    non_cb = set(rdf[~rdf.is_comeback].topic)
    growth = ((last3 - prev3) / (prev3 + 1)).where((last3 >= 50) & df.columns.isin(non_cb))
    if growth.notna().any():
        t = growth.idxmax()
        feed.append(dict(topic=t, tone=C["green"], tag="Momentum watch", title=config.display_name(t),
                         body=f"Fastest riser outside the comeback set: +{growth[t]:.0%} over the previous quarter (sustained growth, not a revival)."))
    drop = ((last3 - prev3) / (prev3 + 1)).where((prev3 >= 50) & df.columns.isin(non_cb))
    if drop.notna().any():
        t = drop.idxmin()
        feed.append(dict(topic=t, tone=C["red"], tag="Decline alert", title=config.display_name(t),
                         body=f"Steepest slide this quarter: {drop[t]:.0%}. Continuous decline, so it does not qualify as a comeback."))
    return feed


def table_grid(rdf: pd.DataFrame, limit: int, key: str) -> None:
    widths = [2.3, 1.7, 1.3, 1.3, 1.4, 1.0]
    with st.container(key="thead"):
        for col, head in zip(st.columns(widths), ["Trend Name", "Previous Peak", "Quiet Duration", "Recent Count", "Priority Status", "Action"]):
            col.markdown(f'<span class="tr-th">{head}</span>', unsafe_allow_html=True)
    for i, (_, r) in enumerate(rdf.head(limit).iterrows()):
        with st.container(key=f"row-{key}-{i}"):
            c = st.columns(widths)
            c[0].markdown(f'<div class="tr-td"><b>{config.display_name(r.topic)}</b></div>', unsafe_allow_html=True)
            c[1].markdown(f'<div class="tr-td tr-mono">{r.historical_peak:,} <span style="color:var(--muted)">· {period_label(r.peak_period)}</span></div>', unsafe_allow_html=True)
            c[2].markdown(f'<div class="tr-td tr-mono">{r.quiet_months} mo</div>', unsafe_allow_html=True)
            c[3].markdown(f'<div class="tr-td tr-mono">{r.recent_count:,.0f}</div>', unsafe_allow_html=True)
            c[4].markdown(f'<div class="tr-td">{badge(r.priority)}</div>', unsafe_allow_html=True)
            c[5].button("Analyze", key=f"act-{key}-{i}", on_click=goto_topic, args=(r.topic,))


def view_overview(df, rdf, stats):
    page_header("Trend Revival Overview", "Social-media comeback intelligence · MapReduce-powered time-series detection")
    cand = rdf[rdf.is_comeback]
    mentions = df.sum(axis=1)
    d_txt, d_cls = pct_delta(mentions.iloc[-6:].sum(), mentions.iloc[-12:-6].sum())
    seen = (df.cumsum() > 0).sum(axis=1)
    tiers = {t: int((cand.priority == t).sum()) for t in ("High", "Medium", "Low")}
    cand_series = df[list(cand.topic)].sum(axis=1) if len(cand) else mentions
    total_posts = stats.get("total_posts", int(mentions.sum()))
    unique_topics = stats.get("unique_topics", df.shape[1])
    cols = st.columns(4)
    cards = [
        ("Total Posts", f"{total_posts:,}", d_txt, d_cls, mentions, C["cyan"]),
        ("Unique Topics", f"{unique_topics:,}", f"+{int(seen.iloc[-1] - seen.iloc[-12]):,} first seen in last 12 months", "up", seen, C["purple"]),
        ("Comeback Candidates", f"{len(cand):,}", f"High {tiers['High']} · Medium {tiers['Medium']} · Low {tiers['Low']}", "up", cand_series, C["green"]),
        ("Analysed Period", f"{len(config.MONTHS)} Months", f"{period_label(config.MONTHS[0])} – {period_label(config.MONTHS[-1])}", "flat", mentions.cumsum(), C["pink"]),
    ]
    for i, (col, card) in enumerate(zip(cols, cards)):
        col.markdown(kpi_card(*card, uid=f"ov{i}"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2.2, 1])
    rmap = {r["topic"]: r for r in rdf.to_dict("records")}
    with left:
        c1, c2 = st.columns([3.2, 1.2])
        options = list(rdf.topic)
        default = list(cand.topic[:4])
        topics = c1.multiselect("Trends on chart", options, default=default, format_func=config.display_name, key="ov_topics")
        normalize = c2.toggle("Normalise to % of peak", value=False, key="ov_norm")
        if topics:
            st.plotly_chart(main_chart(df, rmap, topics, normalize), config={"displayModeBar": False})
        else:
            st.info("Select at least one trend to plot.")
    with right:
        st.markdown('<div class="tr-panel-title">Recent Insights <small>live from detector</small></div>', unsafe_allow_html=True)
        with st.container(key="panel-feed", height=585):
            for i, item in enumerate(build_feed(df, rdf)):
                st.markdown("".join([f'<div class="tr-feed" style="--tone:{item["tone"]}">', f'<div class="tr-feed-tag">{item["tag"]}</div>',
                                     f'<div class="tr-feed-title">{item["title"]}</div>', f'<div class="tr-feed-body">{item["body"]}</div></div>']),
                            unsafe_allow_html=True)
                st.button("View Details", key=f"feed-{i}", on_click=goto_topic, args=(item["topic"],))

    st.markdown('<div class="tr-panel-title" style="margin-top:1.4rem">Comeback Candidates <small>ranked by composite score</small></div>', unsafe_allow_html=True)
    f1, f2 = st.columns([3, 1])
    tier_filter = f1.multiselect("Priority filter", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="ov_tier")
    limit = f2.selectbox("Rows", [8, 12, 20, 27], index=0, key="ov_rows")
    table_grid(cand[cand.priority.isin(tier_filter)], limit, "ov")
    export = cand.drop(columns=[c for c in cand.columns if c.startswith("stage_")])
    st.download_button("Export candidates (CSV)", export.to_csv(index=False).encode(), "comeback_candidates.csv", "text/csv")


def view_timeline(df, rdf):
    page_header("Trend Timeline Explorer", "Inspect any topic against the three detector windows")
    options = list(rdf.topic)
    st.session_state.setdefault("selected_topic", options[0])
    topic = st.selectbox("Topic", options, format_func=config.display_name, key="selected_topic")
    r = rdf[rdf.topic == topic].iloc[0]
    y = df[topic].astype(float)
    k = st.columns(4)
    vals = [("Historical Peak", f"{r.historical_peak:,}", f"{period_label(r.peak_period)}", "flat", y.iloc[:EARLY], C["cyan"]),
            ("Dormancy Coeff", f"{r.dormancy_coeff:.3f}", f"{r.quiet_months} quiet months", "flat", y.iloc[EARLY:], C["purple"]),
            ("Spike Velocity", f"{r.spike_velocity:.0%}", "recent avg vs peak", "up" if r.spike_velocity >= 0.5 else "down", y.iloc[-12:], C["green"]),
            ("Priority", r.priority if r.is_comeback else "None", f"score {r.score:.1f}", "flat", y.rolling(3, min_periods=1).mean(), C["pink"])]
    for i, (col, v) in enumerate(zip(k, vals)):
        col.markdown(kpi_card(*v, uid=f"tl{i}"), unsafe_allow_html=True)
    st.write("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DATES, y=y, name="Monthly mentions", mode="lines+markers", line=dict(color=C["cyan"], width=2.6),
                             marker=dict(size=5), fill="tozeroy", fillcolor="rgba(0,229,255,.07)"))
    fig.add_trace(go.Scatter(x=DATES, y=y.rolling(3, min_periods=1).mean(), name="3-month average", mode="lines",
                             line=dict(color=C["pink"], width=2, dash="dash")))
    add_stage_overlays(fig)
    peak = r.historical_peak
    if peak:
        fig.add_hline(y=peak * config.DETECTOR["dormancy_ratio"], line=dict(color=C["red"], dash="dot", width=1),
                      annotation_text="25% of peak", annotation_font=dict(color=C["red"], size=10), annotation_position="bottom right")
        fig.add_hline(y=peak * config.DETECTOR["recovery_ratio"], line=dict(color=C["green"], dash="dot", width=1),
                      annotation_text="50% of peak", annotation_font=dict(color=C["green"], size=10), annotation_position="top right")
        fig.add_annotation(x=f"{r.peak_period}-01", y=peak, text=f"Peak {peak:,}", showarrow=True, ax=30, ay=-30, arrowcolor=C["cyan"],
                           font=dict(color=C["cyan"]), bgcolor=C["bg"], bordercolor=C["cyan"])
    if r.quiet_months:
        fig.add_vrect(x0=f"{r.quiet_start}-01", x1=f"{r.quiet_end}-01", fillcolor=C["red"], opacity=0.10, line_width=0)
        fig.add_annotation(x=f"{r.quiet_start}-01", y=0, yanchor="top", text=f"Longest valley · {r.quiet_months} mo", showarrow=False,
                           xanchor="left", font=dict(color=C["red"], size=10), yshift=-6)
    st.plotly_chart(style_fig(fig, f"{config.display_name(topic)} · monthly mentions", 460), config={"displayModeBar": False})

    s = st.columns(3)
    stage_defs = [("① Early Peak", r.stage_early_peak, f"Peak {r.historical_peak:,} ≥ noise floor {config.DETECTOR['min_peak']}"),
                  ("② Dormancy Valley", r.stage_dormancy, f"{r.quiet_months} continuous months < 25% of peak (need ≥ {config.DETECTOR['min_quiet_months']})"),
                  ("③ Resurgence", r.stage_resurgence, f"Recent avg = {r.spike_velocity:.0%} of peak (need ≥ 50%)")]
    for col, (name, ok, text) in zip(s, stage_defs):
        mark = '<span class="ok">✔ PASS</span>' if ok else '<span class="bad">✘ FAIL</span>'
        col.markdown(f'<div class="tr-stage"><b>{name}</b> &nbsp; {mark}<p>{text}</p></div>', unsafe_allow_html=True)
    st.caption(f"Verdict: {r.reason}")

    yoy = pd.DataFrame({"month": [m[5:] for m in config.MONTHS], "year": [m[:4] for m in config.MONTHS], "mentions": y.values})
    fig2 = px.bar(yoy, x="month", y="mentions", color="year", barmode="group",
                  color_discrete_sequence=[C["cyan"], C["purple"], C["green"]])
    fig2.update_xaxes(tickvals=[f"{i:02d}" for i in range(1, 13)], ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    st.plotly_chart(style_fig(fig2, "Year-over-year comparison", 340), config={"displayModeBar": False})


def view_analytics(df, rdf):
    page_header("Comeback Detector Analytics", "Tune the three-stage thresholds and re-run detection live on the MapReduce output")
    c = st.columns(3)
    dormancy = c[0].slider("Dormancy ceiling (× peak)", 0.10, 0.40, config.DETECTOR["dormancy_ratio"], 0.05)
    recovery = c[1].slider("Recovery floor (× peak)", 0.30, 1.20, config.DETECTOR["recovery_ratio"], 0.05)
    quiet = c[2].slider("Minimum quiet months", 3, 18, config.DETECTOR["min_quiet_months"], 1)
    live = pd.DataFrame(live_detection(round(dormancy, 2), round(recovery, 2), int(quiet), df))
    cand = live[live.is_comeback]
    tiers = {t: int((cand.priority == t).sum()) for t in ("High", "Medium", "Low")}
    k = st.columns(4)
    cards = [("Candidates", f"{len(cand)}", f"of {len(live):,} topics", "flat", cand.score if len(cand) else [0, 0], C["green"]),
             ("High", f"{tiers['High']}", "ratio ≥ 100%, quiet ≥ 9 mo", "up", [1, 2, tiers["High"] + 1], C["pink"]),
             ("Medium", f"{tiers['Medium']}", "ratio ≥ 75%", "flat", [1, 2, tiers["Medium"] + 1], C["amber"]),
             ("Low", f"{tiers['Low']}", "ratio ≥ recovery floor", "flat", [1, 2, tiers["Low"] + 1], C["cyan"])]
    for i, (col, card) in enumerate(zip(k, cards)):
        col.markdown(kpi_card(*card, uid=f"an{i}"), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.6, 1])
    if len(cand):
        cd = cand.assign(name=cand.topic.map(config.display_name))
        fig = px.scatter(cd, x="quiet_months", y="spike_velocity", size="historical_peak", color="priority", hover_name="name",
                         color_discrete_map=config.PRIORITY_COLORS, size_max=38,
                         category_orders={"priority": ["High", "Medium", "Low"]})
        fig.add_hline(y=1.0, line=dict(color=C["muted"], dash="dot", width=1))
        fig.update_xaxes(title="Quiet months (longest valley)")
        fig.update_yaxes(title="Spike velocity (recent ÷ peak)")
        a.plotly_chart(style_fig(fig, "Dormancy depth vs resurgence strength", 430), config={"displayModeBar": False})
        dist = pd.DataFrame({"priority": list(tiers), "topics": list(tiers.values())})
        fig2 = px.bar(dist, x="priority", y="topics", color="priority", color_discrete_map=config.PRIORITY_COLORS, text="topics")
        fig2.update_traces(textposition="outside", marker_line_width=0)
        b.plotly_chart(style_fig(fig2, "Priority distribution", 430, legend=False), config={"displayModeBar": False})
    else:
        st.warning("No topics satisfy these thresholds - relax the sliders.")
    valid = live[live.stage_dormancy & live.stage_early_peak]
    fig3 = px.histogram(valid, x="spike_velocity", nbins=30, color_discrete_sequence=[C["purple"]])
    fig3.add_vline(x=recovery, line=dict(color=C["green"], dash="dash"), annotation_text="recovery floor", annotation_font=dict(color=C["green"]))
    fig3.update_xaxes(title="Spike velocity among topics that went dormant")
    st.plotly_chart(style_fig(fig3, "Who recovered? Spike-velocity distribution", 340, legend=False), config={"displayModeBar": False})
    show = live.head(60).assign(topic=lambda d: d.topic.map(config.display_name))[
        ["topic", "priority", "score", "historical_peak", "peak_period", "quiet_months", "dormancy_coeff", "recent_count", "spike_velocity", "reason"]]
    st.dataframe(show, hide_index=True, column_config={
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "spike_velocity": st.column_config.NumberColumn("Spike velocity", format="%.2f×")})


def view_compare(df, rdf):
    page_header("Topic Comparison Matrix", "Compare lifecycles side by side: normalised intensity, correlation and trajectory metrics")
    named = [t for t in config.TOPIC_DISPLAY if t in df.columns]
    default = named[:6]
    topics = st.multiselect("Topics (2-10)", list(rdf.topic), default=default, format_func=config.display_name, max_selections=10)
    if len(topics) < 2:
        st.info("Pick at least two topics.")
        return
    norm = df[topics] / df[topics].max()
    labels = [config.display_name(t) for t in topics]
    heat = go.Figure(go.Heatmap(z=norm.T.values, x=DATES, y=labels, colorscale=[[0, C["bg"]], [0.35, "#2b1b5a"], [0.7, C["purple"]], [1, C["cyan"]]],
                                colorbar=dict(title="% of own peak", tickformat=".0%"), hovertemplate="%{y}<br>%{x}<br>%{z:.0%} of peak<extra></extra>"))
    heat.update_layout(hovermode="closest")
    st.plotly_chart(style_fig(heat, "Intensity heatmap (each topic scaled to its own maximum)", 120 + 48 * len(topics), legend=False), config={"displayModeBar": False})
    a, b = st.columns([1, 1.1])
    corr = df[topics].corr()
    cfig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=[[0, C["pink"]], [0.5, C["card"]], [1, C["cyan"]]],
                                text=np.round(corr.values, 2), texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    cfig.update_layout(hovermode="closest")
    a.plotly_chart(style_fig(cfig, "Correlation matrix", 420, legend=False), config={"displayModeBar": False})
    rows = []
    x = np.arange(len(df))
    for t in topics:
        s = df[t].astype(float)
        rows.append({"Topic": config.display_name(t), "Total": int(s.sum()), "Peak": int(s.max()), "Peak month": period_label(s.idxmax()),
                     "Last 3 mo avg": round(s.iloc[-3:].mean(), 1), "Volatility (CV)": round(s.std() / (s.mean() or 1), 2),
                     "Trend slope / mo": round(float(np.polyfit(x, s.values, 1)[0]), 2),
                     "Verdict": rdf.loc[rdf.topic == t, "priority"].iloc[0]})
    with b:
        st.markdown('<div class="tr-panel-title">Trajectory metrics</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        fig = go.Figure()
        for i, t in enumerate(topics):
            fig.add_trace(go.Scatter(x=DATES, y=norm[t] * 100, name=config.display_name(t), mode="lines", line=dict(width=2, color=config.CHART_COLORS[i % 8])))
        st.plotly_chart(style_fig(fig, "Normalised overlay (% of own peak)", 250), config={"displayModeBar": False})


def view_blueprint(df, rdf, stats):
    page_header("Data & Methodology Blueprint", "Architecture, MapReduce jobs, schemas and detection mathematics")
    steps = [("01", "dataset_generator", "124,862 posts · 5 trend signatures", C["cyan"]), ("02", "ingest_data", "typed bulk insert · unique post_id", C["purple"]),
             ("03", "preprocess", "multi-core clean · hashtags · indexes", C["green"]), ("04", "mapreduce_engine", "JS map/reduce → topic_period_counts", C["pink"]),
             ("05", "comeback_detector", "3-window validation · tiers", C["amber"]), ("06", "app.py", "Streamlit + Plotly analytics", C["cyan"])]
    nodes = "".join(f'<div class="tr-node" style="--c:{c}"><span class="n">STEP {n}</span><br><b>{name}</b><small>{desc}</small></div>' for n, name, desc, c in steps)
    st.markdown(f'<div class="tr-flow">{nodes}</div>', unsafe_allow_html=True)

    st.markdown('<div class="tr-panel-title">MapReduce job <small>executed server-side via db.runCommand({mapReduce})</small></div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.code(mapreduce_engine.MAP_JS.strip(), language="javascript")
    m2.code(mapreduce_engine.REDUCE_JS.strip(), language="javascript")

    st.markdown('<div class="tr-panel-title" style="margin-top:1rem">Detection algorithm</div>', unsafe_allow_html=True)
    st.latex(r"\text{Peak}=\max_{m\in[1,12]} c_m,\quad \text{Valley}=\text{longest run } c_m<0.25\cdot\text{Peak},\quad \text{Spike}=\frac{\overline{c}_{34..36}}{\text{Peak}}\ge 0.5")
    st.markdown("**High** = Spike ≥ 100% and quiet ≥ 9 months · **Medium** = Spike ≥ 75% · **Low** = Spike ≥ 50%. "
                "Composite score = 50% resurgence strength + 30% valley length + 20% valley depth.")

    st.markdown('<div class="tr-panel-title" style="margin-top:1rem">Collection schemas</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(config.COLLECTION_SCHEMAS))
    for tab, (name, spec) in zip(tabs, config.COLLECTION_SCHEMAS.items()):
        with tab:
            st.caption(spec["description"] + "  ·  indexes: " + ", ".join(spec["indexes"]))
            st.dataframe(pd.DataFrame({"field": list(spec["fields"]), "type": list(spec["fields"].values())}), hide_index=True)

    st.markdown('<div class="tr-panel-title" style="margin-top:1rem">Runtime</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"item": "MapReduce source", "value": mapreduce_engine.data_source()},
        {"item": "MongoDB connected", "value": str(config.get_db() is not None)},
        {"item": "Posts processed", "value": f"{stats.get('total_posts', 0):,}"},
        {"item": "Topic mentions", "value": f"{stats.get('total_mentions', 0):,}"},
        {"item": "(topic, month) keys", "value": f"{int((df > 0).sum().sum()):,}"},
        {"item": "Evaluated topics", "value": f"{len(rdf):,}"}]), hide_index=True)
    if st.button("Rebuild entire pipeline"):
        with st.status("Rebuilding …", expanded=True) as status:
            run_pipeline(st.write, use_mongo=config.get_db(refresh=True) is not None)
            status.update(label="Pipeline rebuilt", state="complete")
        st.cache_data.clear()
        st.rerun()


# =========================================================================== #
# Entry point
# =========================================================================== #
def main() -> None:
    inject_css()
    st.session_state.setdefault("nav", NAV[0])
    with st.sidebar:
        st.markdown('<div class="tr-brand"><div class="tr-logo">↺</div><div><div class="tr-brand-name">Trend Revival</div>'
                    '<div class="tr-brand-sub">Comeback Analytics</div></div></div><div class="tr-side-label">Navigation</div>', unsafe_allow_html=True)
        st.radio("Navigation", NAV, key="nav", label_visibility="collapsed", format_func=lambda x: f"{NAV_ICONS[x]}   {x}")
    ensure_pipeline()
    df, results, stats = load_data(0)
    rdf = pd.DataFrame(results)
    with st.sidebar:
        mongo = mapreduce_engine.data_source().startswith("MongoDB")
        tone = C["green"] if mongo else C["amber"]
        st.markdown(f'<div class="tr-side-label">Engine</div><div class="tr-status"><span class="tr-dot" style="color:{tone};background:{tone}"></span>'
                    f'{mapreduce_engine.data_source()}<br><span style="font-size:.7rem">{df.shape[1]:,} topics · {len(df)} months</span></div>', unsafe_allow_html=True)
    view = st.session_state["nav"]
    if view == NAV[0]:
        view_overview(df, rdf, stats)
    elif view == NAV[1]:
        view_timeline(df, rdf)
    elif view == NAV[2]:
        view_analytics(df, rdf)
    elif view == NAV[3]:
        view_compare(df, rdf)
    else:
        view_blueprint(df, rdf, stats)


main()
