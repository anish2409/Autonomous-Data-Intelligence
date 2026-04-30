"""
dashboard/app1.py  ·  ADI System — Production-Ready Build
──────────────────────────────────────────────────────────
FIXES in this build:
  BUG-1  Duplicate/dead code after return in detect_anomalies_ml (lines 78-81)
  BUG-2  selectbox rendered in sidebar BEFORE mode is set → crashes on non-CSV runs
  BUG-3  kpi_df / anomaly_df overwritten AFTER CSV override → CSV data lost
  BUG-4  Live Stream view calls ActionEngine / AIDatabase without guard → crash
  BUG-5  AI Council view double-button (render_ai_council_section + run_council)
         duplicate run logic + broken anomaly_df re-fetch
  BUG-6  langgraph_agent import path wrong ('agents.langgraph_agent' vs root)
  BUG-7  _causal_narrative & _agent_causal use old static causal_map (not dynamic)
  BUG-8  Causal Analysis view references causal_df before assignment in else branch
  BUG-9  _csv_file_name never written to session_state on file upload
  BUG-10 Mode radio placed after anomaly method selectbox → wrong render order
CONNECTED: langgraph_agent.py  (AgentState keys: anomaly_report / business_report /
           final_decision) wired into run_ai_council() as primary path.
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, sys, re, time, random, queue, threading, sqlite3
from datetime import datetime, timedelta, timezone
from typing import TypedDict, List, Dict, Any, Optional

# ─── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ADI System Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SESSION-STATE SAFETY NET  (idempotent — runs every rerun)
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS: dict = {
    "_csv_file_name": "LIVE CSV",
    "_csv_proc":      None,
    "_csv_kpi_df":    pd.DataFrame(),
    "_csv_anom_df":   pd.DataFrame(),
    "_csv_insight":   "",
    "_csv_file_key":  "",
    "stream_logs":    [],
    "stream_alerts":  0,
    "anomaly_df":     pd.DataFrame(),
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# 2.  OPTIONAL IMPORTS — LangGraph agent + broker + DB
# ══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

streaming_available = False
agent_app           = None
broker              = None
ActionEngine        = None
AIDatabase          = None

try:
    from local_broker import LocalKafka                       # type: ignore
    broker = LocalKafka()
    streaming_available = True
except ImportError:
    pass

try:
    # Support both 'agents/langgraph_agent.py' and root-level 'langgraph_agent.py'
    try:
        from agents.langgraph_agent import app as agent_app   # type: ignore
    except ImportError:
        from langgraph_agent import app as agent_app           # type: ignore
    streaming_available = True
except ImportError:
    pass

try:
    from db_manager import AIDatabase as _AID                 # type: ignore
    AIDatabase = _AID
except ImportError:
    pass

try:
    from autonomous_action import ActionEngine as _AE         # type: ignore
    ActionEngine = _AE
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREMIUM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ── background ────────────────────────────────────────────────────────────── */
.stApp {
  background:
    radial-gradient(ellipse at 18% 0%,  #0f1d3e 0%, transparent 55%),
    radial-gradient(ellipse at 82% 90%, #1a0a2e 0%, transparent 50%),
    linear-gradient(180deg, #050a18 0%, #04070f 100%);
}
.stApp::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.35;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
}

/* ── sidebar ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,rgba(10,15,32,.98) 0%,rgba(6,10,22,.99) 100%);
  border-right: 1px solid rgba(124,58,237,.15);
  backdrop-filter: blur(24px);
}

/* ── hero header ───────────────────────────────────────────────────────────── */
.adi-hero {
  background: linear-gradient(135deg,rgba(20,16,52,.7) 0%,rgba(10,16,36,.8) 100%);
  border:1px solid rgba(124,58,237,.2); border-radius:20px;
  padding:2rem 2.4rem 1.6rem; margin-bottom:1.6rem;
  position:relative; overflow:hidden; backdrop-filter:blur(20px);
}
.adi-hero::before {
  content:''; position:absolute; top:-60px; left:-60px;
  width:250px; height:250px;
  background:radial-gradient(circle,rgba(124,58,237,.12) 0%,transparent 70%);
  animation:heroGlow 6s ease-in-out infinite alternate;
}
@keyframes heroGlow {
  from { transform:scale(1) translate(0,0);    opacity:.6; }
  to   { transform:scale(1.3) translate(20px,15px); opacity:1; }
}
.adi-hero h1 {
  font-size:2.1rem; font-weight:800; margin:0; letter-spacing:-.03em;
  background:linear-gradient(135deg,#f1f5f9 30%,#a78bfa 80%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  position:relative; z-index:1;
}
.adi-hero .hero-sub { font-size:.8rem; color:#475569; margin:.4rem 0 0; position:relative; z-index:1; }
.adi-hero .hero-badges { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1rem; position:relative; z-index:1; }
.hero-badge { display:inline-flex; align-items:center; gap:.3rem; font-size:.68rem; font-weight:700;
  letter-spacing:.1em; text-transform:uppercase; border-radius:99px; padding:.22rem .7rem; }
.hero-badge-violet { background:rgba(124,58,237,.15); border:1px solid rgba(124,58,237,.35); color:#a78bfa; }
.hero-badge-blue   { background:rgba(59,130,246,.12);  border:1px solid rgba(59,130,246,.3);  color:#60a5fa; }
.hero-badge-green  { background:rgba(34,197,94,.10);   border:1px solid rgba(34,197,94,.3);   color:#22c55e; }
.hero-badge-orange { background:rgba(249,115,22,.10);  border:1px solid rgba(249,115,22,.3);  color:#fb923c; }

/* ── metric cards ───────────────────────────────────────────────────────────── */
.kpi-glass {
  background:linear-gradient(135deg,rgba(22,24,56,.65) 0%,rgba(14,18,42,.8) 100%);
  border:1px solid rgba(124,58,237,.18); border-radius:18px;
  padding:1.5rem 1.8rem;
  box-shadow:0 4px 24px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.04);
  backdrop-filter:blur(16px); position:relative; overflow:hidden;
  transition:all .3s ease;
}
.kpi-glass::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(124,58,237,.6),transparent);
}
.kpi-glass:hover { border-color:rgba(124,58,237,.42); transform:translateY(-3px);
  box-shadow:0 8px 36px rgba(124,58,237,.18), inset 0 1px 0 rgba(255,255,255,.07); }
.kpi-icon  { font-size:1.6rem; margin-bottom:.4rem; opacity:.9; }
.kpi-label { font-size:.7rem; letter-spacing:.13em; text-transform:uppercase;
  color:#475569; font-weight:700; margin-bottom:.3rem; }
.kpi-value { font-size:1.95rem; font-weight:800; color:#f1f5f9; line-height:1; letter-spacing:-.02em; }
.kpi-delta-pos { font-size:.76rem; color:#22c55e; font-weight:700; margin-top:.4rem; }
.kpi-delta-neg { font-size:.76rem; color:#ef4444; font-weight:700; margin-top:.4rem; }
.kpi-delta-neu { font-size:.76rem; color:#64748b; font-weight:600; margin-top:.4rem; }

/* ── agent box ──────────────────────────────────────────────────────────────── */
.agent-box {
  background:linear-gradient(135deg,rgba(6,9,20,.95) 0%,rgba(10,15,30,.95) 100%);
  border-radius:10px; padding:1.2rem 1.4rem;
  font-family:'DM Mono',monospace; font-size:.81rem; line-height:1.75;
  margin:.5rem 0; border:1px solid rgba(124,58,237,.12);
  border-top:1px solid rgba(124,58,237,.25); color:#94a3b8; white-space:pre-wrap;
}

/* ── council bar ────────────────────────────────────────────────────────────── */
.council-bar {
  display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;
  background:rgba(10,15,32,.7); border:1px solid rgba(124,58,237,.12);
  border-radius:12px; padding:.7rem 1.1rem; margin:.8rem 0 1rem; font-size:.78rem;
}
.council-agent { display:flex; align-items:center; gap:.35rem; color:#64748b; font-weight:600; }
.council-agent .dot {
  width:6px; height:6px; border-radius:50%; background:#22c55e;
  box-shadow:0 0 6px #22c55e; animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.5; transform:scale(.7); }
}

/* ── causal card ────────────────────────────────────────────────────────────── */
.causal-card {
  background:linear-gradient(135deg,rgba(10,14,30,.9) 0%,rgba(16,22,44,.9) 100%);
  border:1px solid rgba(124,58,237,.14); border-left:3px solid #7c3aed;
  border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:.7rem;
  font-family:'DM Mono',monospace; font-size:.8rem; line-height:1.7; color:#94a3b8;
  transition:border-left-color .2s ease;
}
.causal-card:hover { border-left-color:#a78bfa; }
.causal-card .cc-metric { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;
  color:#7c3aed; font-weight:700; margin-bottom:.3rem; }
.causal-card .cc-ate  { font-size:1.2rem; font-weight:800; color:#f1f5f9; letter-spacing:-.01em; }
.causal-card .cc-conf { font-size:.72rem; color:#22c55e; font-weight:600; }
.causal-card .cc-hyp  { font-size:.79rem; color:#94a3b8; margin-top:.4rem; font-style:italic; }

/* ── misc UI ────────────────────────────────────────────────────────────────── */
.section-header {
  font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:#7c3aed; border-bottom:1px solid rgba(124,58,237,.15);
  padding-bottom:.55rem; margin:1.6rem 0 1rem; display:flex; align-items:center; gap:.5rem;
}
.ai-insight {
  background:linear-gradient(135deg,rgba(124,58,237,.10) 0%,rgba(59,130,246,.07) 100%);
  border:1px solid rgba(124,58,237,.25); border-radius:12px;
  padding:1rem 1.4rem; margin:.8rem 0 1.2rem; font-size:.88rem; color:#c4b5fd; line-height:1.65;
}
.ai-insight::before { content:'✦ AI INSIGHT'; display:block; font-size:.62rem;
  letter-spacing:.16em; color:#7c3aed; font-weight:700; margin-bottom:.35rem; }
.empty-state { text-align:center; padding:3rem 2rem; color:#334155; }
.empty-state .es-icon  { font-size:3rem; margin-bottom:1rem; opacity:.6; }
.empty-state .es-title { font-size:1.1rem; font-weight:700; color:#475569; margin-bottom:.5rem; }
.empty-state .es-body  { font-size:.85rem; color:#334155; line-height:1.6; }
.csv-badge {
  display:inline-flex; align-items:center; gap:.4rem;
  background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.3);
  border-radius:99px; padding:.25rem .75rem;
  font-size:.72rem; font-weight:700; color:#22c55e; letter-spacing:.08em; text-transform:uppercase;
}
.upload-zone {
  background:linear-gradient(135deg,rgba(124,58,237,.07) 0%,rgba(59,130,246,.05) 100%);
  border:1.5px dashed rgba(124,58,237,.3); border-radius:14px;
  padding:1.6rem; text-align:center; margin:.8rem 0; transition:border-color .2s ease;
}
.upload-zone:hover { border-color:rgba(124,58,237,.55); }
@keyframes fadeSlideUp {
  from { opacity:0; transform:translateY(14px); }
  to   { opacity:1; transform:translateY(0); }
}
.fade-in   { animation:fadeSlideUp .45s ease both; }
.fade-in-1 { animation-delay:.04s; } .fade-in-2 { animation-delay:.10s; }
.fade-in-3 { animation-delay:.17s; } .fade-in-4 { animation-delay:.25s; }
.severity-CRITICAL { color:#ef4444; font-weight:700; letter-spacing:.06em; }
.severity-HIGH     { color:#f97316; font-weight:700; letter-spacing:.06em; }
.severity-MEDIUM   { color:#eab308; font-weight:700; letter-spacing:.06em; }
.severity-LOW      { color:#22c55e; font-weight:700; letter-spacing:.06em; }
/* Streamlit overrides */
div[data-testid="stMetricValue"] { font-family:'Syne',sans-serif; font-weight:800; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:rgba(10,15,32,.6); border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"]      { border-radius:8px; font-size:.82rem; font-weight:600; }
div[data-testid="stDataFrame"]    { border-radius:12px; overflow:hidden;
  border:1px solid rgba(124,58,237,.12) !important; }
.stSelectbox>div, .stMultiSelect>div {
  background:rgba(10,15,32,.7) !important; border-radius:10px !important;
  border-color:rgba(124,58,237,.22) !important; }
.stFileUploader>div {
  background:rgba(10,15,32,.5) !important; border-radius:12px !important;
  border-color:rgba(124,58,237,.22) !important; }
.stButton>button { border-radius:10px !important; font-family:'Syne',sans-serif !important;
  font-weight:700 !important; letter-spacing:.04em !important; transition:all .2s ease !important; }
.stButton>button[kind="primary"] {
  background:linear-gradient(135deg,#7c3aed,#6d28d9) !important;
  border:1px solid rgba(124,58,237,.4) !important;
  box-shadow:0 4px 16px rgba(124,58,237,.3) !important; }
.stButton>button[kind="primary"]:hover {
  box-shadow:0 6px 24px rgba(124,58,237,.45) !important; transform:translateY(-1px) !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(10,15,32,.4); border-radius:99px; }
::-webkit-scrollbar-thumb { background:rgba(124,58,237,.35); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:rgba(124,58,237,.6); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_card(icon, label, value, delta="", delta_sign=0, delay=1):
    sign_class = "kpi-delta-pos" if delta_sign > 0 else ("kpi-delta-neg" if delta_sign < 0 else "kpi-delta-neu")
    arrow      = "↑ " if delta_sign > 0 else ("↓ " if delta_sign < 0 else "→ ")
    delta_html = f'<div class="{sign_class}">{arrow}{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-glass fade-in fade-in-{delay}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def render_empty_state(icon="📡", title="No data yet",
                       body="Upload a dataset to begin intelligence analysis."):
    st.markdown(f"""
    <div class="empty-state fade-in">
      <div class="es-icon">{icon}</div>
      <div class="es-title">{title}</div>
      <div class="es-body">{body}</div>
    </div>""", unsafe_allow_html=True)


def render_insight(text: str):
    st.markdown(f'<div class="ai-insight fade-in">{text}</div>', unsafe_allow_html=True)


def render_csv_badge(filename, rows, cols):
    st.markdown(f"""
    <div style="margin:.3rem 0 .8rem;">
      <span class="csv-badge">● LIVE DATA</span>&nbsp;
      <span style="font-size:.75rem;color:#64748b;">
        {filename} &nbsp;·&nbsp; {rows:,} rows &nbsp;·&nbsp; {cols} cols
      </span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ML / ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies_ml(df: pd.DataFrame) -> pd.DataFrame:
    """IsolationForest — returns df with ml_anomaly column (-1 = anomaly)."""
    if df is None or df.empty:
        return df
    df_copy    = df.copy()
    numeric_df = df_copy.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median()).dropna(axis=1, how="all")
    if numeric_df.empty or numeric_df.shape[1] == 0:
        df_copy["ml_anomaly"] = 1
        return df_copy
    preds = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit_predict(numeric_df)
    df_copy = df_copy.loc[numeric_df.index].copy()
    df_copy["ml_anomaly"] = preds
    return df_copy


def detect_anomalies_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score anomaly detection — returns df with z_score + severity columns."""
    if df is None or df.empty:
        return df
    df_copy     = df.copy()
    numeric_cols = df_copy.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return df_copy
    z_scores = (df_copy[numeric_cols] - df_copy[numeric_cols].mean()) / (df_copy[numeric_cols].std() + 1e-9)
    df_copy["z_score"] = z_scores.abs().max(axis=1)
    df_copy["severity"] = df_copy["z_score"].apply(
        lambda z: "CRITICAL" if z > 3 else ("HIGH" if z > 2 else ("MEDIUM" if z > 1 else "LOW"))
    )
    return df_copy


def compute_causal_impact(kpi_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Correlation-proxy causal impact for interactive Causal Analysis view."""
    if kpi_df is None or kpi_df.empty:
        return pd.DataFrame()
    numeric_df = kpi_df.select_dtypes(include="number")
    if target_col not in numeric_df.columns:
        return pd.DataFrame()
    corr = numeric_df.corr()[target_col].drop(target_col)
    out  = corr.reset_index()
    out.columns = ["cause_variable", "ate"]
    out["confidence"]      = out["ate"].abs()
    out["effect_variable"] = target_col
    out["method"]          = "Correlation Proxy"
    out["explanation"]     = "Derived from Pearson correlation with target metric."
    out["anomaly_id"]      = range(1, len(out) + 1)
    return out.sort_values("confidence", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  AUTO CSV PROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower()).strip("_")


def _detect_col_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series): return "date"
    if pd.api.types.is_numeric_dtype(series):        return "numeric"
    return "categorical"


def _infer_dataset_type(df: pd.DataFrame, col_meta: dict) -> str:
    cols = " ".join(df.columns.tolist()).lower()
    if any(k in cols for k in ["revenue", "sales", "amount", "price", "order"]): return "sales"
    if any(k in cols for k in ["log", "level", "severity", "error", "event"]):   return "logs"
    if any(k in cols for k in ["cpu", "memory", "latency", "rps", "metric"]):    return "metrics"
    if any(k in cols for k in ["user", "session", "click", "conversion"]):        return "analytics"
    return "general"


def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    cols = set(df.columns)
    for p in [c for c in cols if "price" in c]:
        for q in [c for c in cols if "quantity" in c or "qty" in c]:
            if "revenue" not in cols and "total" not in cols:
                try: df["auto_revenue"] = df[p] * df[q]
                except Exception: pass
    rev  = [c for c in cols if any(k in c for k in ["revenue", "sales", "amount"])]
    cost = [c for c in cols if "cost" in c]
    if rev and cost and "profit_margin" not in cols:
        try:
            df["auto_profit_margin"] = ((df[rev[0]] - df[cost[0]]) / df[rev[0]]).replace([np.inf, -np.inf], np.nan)
        except Exception: pass
    return df


def auto_process_csv(raw: pd.DataFrame) -> dict:
    df = raw.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    col_meta: dict = {}
    for col in df.columns:
        kind = _detect_col_kind(df[col])
        col_meta[col] = kind
        if kind == "date":
            try: df[col] = pd.to_datetime(df[col], infer_datetime_format=True, utc=True)
            except Exception: pass
        elif kind == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, kind in col_meta.items():
        if kind == "numeric":    df[col] = df[col].fillna(df[col].median())
        elif kind == "categorical": df[col] = df[col].fillna("Unknown")
    df = _derive_metrics(df)
    dataset_type = _infer_dataset_type(df, col_meta)
    return {
        "df": df, "col_meta": col_meta, "dataset_type": dataset_type,
        "row_count": len(df), "col_count": len(df.columns),
        "date_cols":    [c for c, k in col_meta.items() if k == "date"],
        "numeric_cols": [c for c, k in col_meta.items() if k == "numeric"],
        "cat_cols":     [c for c, k in col_meta.items() if k == "categorical"],
    }


def generate_csv_insight(proc: dict) -> str:
    df, num, cats, ds = proc["df"], proc["numeric_cols"], proc["cat_cols"], proc["dataset_type"]
    lines = []
    if num:
        primary = num[0]; series = df[primary].dropna()
        if len(series) > 5:
            z = (series - series.mean()) / (series.std() + 1e-9)
            sc = int((z.abs() > 2.5).sum())
            if sc:
                lines.append(f"**{sc} spike(s)** in `{primary}` (z>2.5σ) — investigate anomalies.")
    for cat in cats[:2]:
        top_val   = df[cat].value_counts().idxmax()
        top_share = df[cat].value_counts(normalize=True).iloc[0]
        if top_share > 0.55:
            lines.append(f"`{cat}` dominated by **{top_val}** ({top_share:.0%}) — skewed distribution.")
    type_hints = {
        "sales":     "Dataset is **sales/revenue** data. KPI and anomaly views pre-loaded.",
        "logs":      "Dataset resembles **event logs**. Filter by severity for anomaly detection.",
        "metrics":   "Dataset looks like **system metrics**. Time-series trends charted below.",
        "analytics": "Dataset contains **user analytics** signals.",
        "general":   "Dataset schema is **general-purpose**. Numeric columns mapped to KPI cards.",
    }
    lines.append(type_hints.get(ds, ""))
    return "  \n".join(l for l in lines if l)


def _csv_to_kpi_df(proc: dict) -> pd.DataFrame:
    df, date_cols, num_cols, cat_cols = proc["df"], proc["date_cols"], proc["numeric_cols"], proc["cat_cols"]
    out = df.copy()
    if "snapshot_ts" not in out.columns:
        if date_cols: out = out.rename(columns={date_cols[0]: "snapshot_ts"})
        else: out["snapshot_ts"] = pd.date_range(end=datetime.now(timezone.utc), periods=len(out), freq="h")
    if "region" not in out.columns:
        rc = [c for c in cat_cols if c != "snapshot_ts"]
        out = out.rename(columns={rc[0]: "region"}) if rc else (out.__setitem__("region", "Default") or out)
    if "total_revenue" not in out.columns:
        rc = [c for c in num_cols if any(k in c for k in ["revenue","sales","amount","total","value","auto_revenue"])]
        if rc:   out = out.rename(columns={rc[0]: "total_revenue"})
        elif num_cols: out = out.rename(columns={num_cols[0]: "total_revenue"})
        else:    out["total_revenue"] = 0.0
    if "total_orders" not in out.columns:
        rc = [c for c in num_cols if any(k in c for k in ["order","count","qty"])]
        out = out.rename(columns={rc[0]: "total_orders"}) if rc else (out.__setitem__("total_orders", 1) or out)
    for col in ["return_rate", "avg_order_val"]:
        if col not in out.columns: out[col] = np.nan
    if "category" not in out.columns: out["category"] = proc["dataset_type"].title()
    keep = ["snapshot_ts","region","category","total_revenue","total_orders","return_rate","avg_order_val"]
    for c in keep:
        if c not in out.columns: out[c] = np.nan
    return out[keep]


def _csv_to_anomaly_df(proc: dict) -> pd.DataFrame:
    df, num_cols = proc["df"], proc["numeric_cols"]
    rows = []
    for col in num_cols[:4]:
        series = df[col].dropna()
        if len(series) < 6: continue
        mu, sigma = series.mean(), series.std()
        if sigma < 1e-9: continue
        zs = (series - mu) / sigma
        for idx, z in zs[zs.abs() > 2.5].items():
            sev = "CRITICAL" if abs(z) > 4 else ("HIGH" if abs(z) > 3 else "MEDIUM")
            region = df.loc[idx, proc["cat_cols"][0]] if proc["cat_cols"] and idx in df.index else "N/A"
            rows.append({
                "anomaly_id": len(rows)+1, "metric_name": col,
                "metric_value": float(series[idx]), "expected_value": float(mu),
                "z_score": float(z), "severity": sev, "region": str(region),
                "category": proc["dataset_type"].title(),
                "detected_at": datetime.now(timezone.utc),
            })
        if len(rows) >= 50: break
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "anomaly_id","metric_name","metric_value","expected_value",
        "z_score","severity","region","category","detected_at"])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LIVE STREAM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_STREAM_QUEUE_KEY = "_adi_stream_queue"
_SEV_COLORS = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
_TYPE_ICONS = {"revenue_spike":"💰","anomaly_detected":"🚨","order_surge":"📦",
               "return_alert":"↩️","model_decision":"🧠","pipeline_tick":"⚙️","schema_drift":"🔧"}


def _get_stream_queue():
    if _STREAM_QUEUE_KEY not in st.session_state:
        st.session_state[_STREAM_QUEUE_KEY] = queue.Queue(maxsize=200)
    return st.session_state[_STREAM_QUEUE_KEY]


def _broker_feed_thread(q, stop_event):
    """Daemon: real broker if available, else synthetic events."""
    _rng = random.Random(int(time.time()))
    _kinds     = ["revenue_spike","anomaly_detected","order_surge","return_alert","model_decision","pipeline_tick","schema_drift"]
    _regions   = ["North","South","East","West"]
    _severities= ["LOW","LOW","MEDIUM","HIGH","CRITICAL"]
    _metrics   = ["total_revenue","return_rate","avg_order_val","total_orders"]
    while not stop_event.is_set():
        try:
            ev = {"type": _rng.choice(_kinds), "region": _rng.choice(_regions),
                  "metric": _rng.choice(_metrics), "value": round(_rng.uniform(800, 98000), 2),
                  "severity": _rng.choice(_severities),
                  "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if not q.full(): q.put_nowait(ev)
            time.sleep(_rng.uniform(0.8, 2.2))
        except Exception:
            time.sleep(2)


def _ensure_stream_running():
    if st.session_state.get("_stream_thread_running"): return
    q = _get_stream_queue(); stop_ev = threading.Event()
    threading.Thread(target=_broker_feed_thread, args=(q, stop_ev), daemon=True, name="adi-stream").start()
    st.session_state["_stream_thread_running"] = True
    st.session_state["_stream_stop_event"]     = stop_ev
    st.session_state["_stream_log"]            = []


def _drain_queue_to_log(max_drain: int = 30):
    q = _get_stream_queue(); log = st.session_state.get("_stream_log", []); n = 0
    while n < max_drain:
        try: log.append(q.get_nowait()); n += 1
        except queue.Empty: break
    st.session_state["_stream_log"] = log[-150:]
    return st.session_state["_stream_log"]


def render_live_stream_section():
    _ensure_stream_running()
    st.markdown('<div class="section-header">⚡ Live AI Stream</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 4])
    auto_refresh = c1.toggle("Auto-refresh", value=st.session_state.get("_stream_auto_refresh_val", True),
                             key="_stream_auto_refresh")
    st.session_state["_stream_auto_refresh_val"] = auto_refresh

    manual_refresh = c2.button("⟳ Refresh", key="_stream_manual_refresh")

    # Drain queue BEFORE rendering so events always show
    if manual_refresh or auto_refresh:
        _drain_queue_to_log()

    log = st.session_state.get("_stream_log", [])
    c3.caption(f"{'🟢 Streaming' if auto_refresh else '⏸ Paused'} · {len(log)} events buffered")

    if not log:
        render_empty_state("📡", "Awaiting first event…",
                           "Stream thread is running — events appear in 1-2 seconds. Hit ⟳ Refresh.")
    else:
        rows_html = ""
        for ev in reversed(log[-60:]):
            sev    = ev.get("severity", "LOW")
            color  = _SEV_COLORS.get(sev, "#64748b")
            icon   = _TYPE_ICONS.get(ev.get("type", ""), "●")
            ts     = ev.get("ts", "")
            kind   = ev.get("type", "event").replace("_", " ").upper()
            val    = ev.get("value", "")
            region = ev.get("region", "")
            metric = ev.get("metric", "")
            val_str = (f'<span style="color:#a78bfa;font-weight:600;">{val:,.2f}</span>'
                       if isinstance(val, (int, float)) else str(val))
            meta = " · ".join(filter(None, [region, metric]))
            rows_html += f"""
            <div style="display:flex;align-items:flex-start;gap:.8rem;padding:.55rem .8rem;
              border-radius:8px;border-left:3px solid {color};background:rgba(15,20,40,.55);
              margin-bottom:4px;font-family:'DM Mono',monospace;font-size:.78rem;">
              <span style="font-size:1rem;flex-shrink:0;">{icon}</span>
              <span style="color:{color};font-weight:700;width:140px;flex-shrink:0;">{kind}</span>
              <span style="color:#94a3b8;flex:1;">{meta}</span>
              <span>{val_str}</span>
              <span style="color:#334155;font-size:.7rem;white-space:nowrap;margin-left:.5rem;">
                {ts[11:19] if len(ts) > 18 else ts}
              </span>
            </div>"""
        st.markdown(f"""
        <div style="max-height:360px;overflow-y:auto;
          background:linear-gradient(135deg,rgba(8,12,24,.85) 0%,rgba(12,18,36,.85) 100%);
          border:1px solid rgba(124,58,237,.18);border-radius:12px;padding:.6rem .7rem;">
          {rows_html}
        </div>""", unsafe_allow_html=True)

    # Schedule next rerun AFTER all rendering is done
    if auto_refresh:
        time.sleep(2)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AI COUNCIL — 3-AGENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class CouncilState(TypedDict):
    anomalies:       List[Dict[str, Any]]
    kpi_summary:     Dict[str, Any]
    analyst_output:  str
    causal_output:   str
    decision_output: str
    confidence:      float
    final_action:    str
    priority:        str
    completed:       bool


def _agent_analyst(state: dict) -> dict:
    anomalies = state.get("anomalies", [])
    kpi       = state.get("kpi_summary", {})

    if not anomalies:
        state["analyst_output"] = (
            "✅ NO ANOMALIES DETECTED\n\n"
            f"  Revenue baseline : ${kpi.get('total_revenue', 0):>12,.0f}\n"
            f"  Total orders     : {kpi.get('total_orders', 0):>12,.0f}\n"
            f"  Return rate      : {kpi.get('return_rate', 0):>11.1%}\n\n"
            "  System operating within normal parameters.\n"
            "  Recommendation: Continue standard monitoring."
        )
        state["confidence"] = 0.92
        return state

    top  = sorted(anomalies, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:5]
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  ANOMALY EXPERT — DETECTION REPORT",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
    ]
    for i, a in enumerate(top, 1):
        z    = abs(a.get("z_score", 0))
        met  = a.get("metric_name", "unknown")
        sev  = a.get("severity", "MEDIUM")
        reg  = a.get("region", "N/A")
        val  = a.get("metric_value", 0)
        exp  = a.get("expected_value", 0)
        diff = val - exp
        lines += [
            f"  [{i}] {sev} — {met.upper()}",
            f"      Region   : {reg}",
            f"      Observed : {val:,.2f}",
            f"      Expected : {exp:,.2f}",
            f"      Delta    : {diff:+,.2f}  (z={z:.2f}σ)",
            f"      Signal   : {'🔴 CRITICAL BREACH' if z > 4 else ('🟠 HIGH ALERT' if z > 3 else '🟡 ELEVATED')}",
            "",
        ]

    max_z  = max(abs(a.get("z_score", 0)) for a in top)
    conf   = min(0.99, 0.60 + max_z * 0.04)
    lines += [
        f"  SUMMARY: {len(anomalies)} total anomalies detected.",
        f"  Top z-score: {max_z:.2f}σ   Confidence: {conf:.0%}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    state["analyst_output"] = "\n".join(lines)
    state["confidence"]     = conf
    return state


def _agent_causal(state: dict) -> dict:
    anomalies = state.get("anomalies", [])
    kpi       = state.get("kpi_summary", {})

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  CAUSAL INFERENCE ENGINE — ATE ANALYSIS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
    ]

    if not anomalies:
        lines.append("  No anomalies to analyze. Causal chain not triggered.")
        state["causal_output"] = "\n".join(lines)
        return state

    seen: set = set()
    for a in sorted(anomalies, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:5]:
        met = a.get("metric_name", "")
        if met in seen:
            continue
        seen.add(met)

        val  = float(a.get("metric_value", 0))
        exp  = float(a.get("expected_value", 0))
        ate  = val - exp
        z    = abs(float(a.get("z_score", 0)))
        conf = min(0.99, 0.50 + z * 0.07)
        pct  = (ate / max(abs(exp), 1e-9)) * 100
        narr = _causal_narrative(met, ate, z, kpi)

        lines += [
            f"  METRIC  : {met.upper()}",
            f"  ATE     : {ate:+,.4f}   ({pct:+.1f}% from baseline)",
            f"  z-score : {z:.2f}σ      Confidence: {conf:.0%}",
            f"  CAUSE   : {narr}",
            "",
        ]

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    state["causal_output"] = "\n".join(lines)
    return state


def _agent_decision(state: dict) -> dict:
    anomalies = state.get("anomalies", [])
    conf      = state.get("confidence", 0.0)

    if not anomalies:
        state["decision_output"] = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  EXECUTIVE REVIEWER — VERDICT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "  VERDICT  : ✅ APPROVE — No action required\n"
            "  PRIORITY : P4 (Routine)\n"
            "  AGENTS   : 3/3 in agreement\n\n"
            "  System nominal. Continue standard ops.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        state["final_action"] = "No action — system nominal."
        state["priority"]     = "P4"
        state["completed"]    = True
        return state

    critical = [a for a in anomalies if a.get("severity") == "CRITICAL"]
    high     = [a for a in anomalies if a.get("severity") == "HIGH"]
    medium   = [a for a in anomalies if a.get("severity") == "MEDIUM"]

    if critical:
        priority = "P1"
        top      = critical[0]
        verdict  = "🔴 REJECT — IMMEDIATE ACTION"
        action   = (
            f"FREEZE promotions for {top.get('metric_name','metric')} in {top.get('region','region')}.\n"
            f"  • Activate circuit breaker — halt automated discounting.\n"
            f"  • Alert on-call ops team within 15 minutes.\n"
            f"  • Rollback last deployment if revenue anomaly persists > 30min.\n"
            f"  • Escalate to P1 incident channel."
        )
    elif high:
        priority = "P2"
        top      = high[0]
        verdict  = "🟠 FLAG — URGENT INVESTIGATION"
        action   = (
            f"INVESTIGATE {top.get('metric_name','metric')} spike in {top.get('region','region')}.\n"
            f"  • Pull last 6h of raw transaction logs.\n"
            f"  • Check payment gateway status and error rates.\n"
            f"  • Review recent A/B test variants for pricing leaks.\n"
            f"  • Schedule 30-min team review within 2 hours."
        )
    elif medium:
        priority = "P3"
        verdict  = "🟡 MONITOR — ELEVATED WATCH"
        action   = (
            "Increase anomaly sampling to 1-minute intervals for 4 hours.\n"
            "  • Set alert threshold to 1.5σ for affected metrics.\n"
            "  • Review trend in next scheduled standup.\n"
            "  • No immediate escalation required."
        )
    else:
        priority = "P4"
        verdict  = "🟢 APPROVE — ROUTINE"
        action   = "Standard monitoring. No escalation needed."

    n_crit = len(critical); n_high = len(high); n_med = len(medium)
    state["decision_output"] = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  EXECUTIVE REVIEWER — VERDICT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"  VERDICT    : {verdict}\n"
        f"  PRIORITY   : {priority}\n"
        f"  CONFIDENCE : {conf:.0%}   AGENTS: 3/3 consensus\n\n"
        f"  ANOMALY BREAKDOWN:\n"
        f"    Critical: {n_crit}  |  High: {n_high}  |  Medium: {n_med}\n\n"
        f"  RECOMMENDED ACTION:\n"
        f"  {action}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    state["final_action"] = action
    state["priority"]     = priority
    state["completed"]    = True
    return state


def _causal_narrative(met: str, ate: float, z: float, kpi: dict) -> str:
    direction = "spike" if ate > 0 else "drop"
    magnitude = "EXTREME" if abs(z) > 4 else ("HIGH" if abs(z) > 3 else "MODERATE")
    hypotheses = {
        "total_revenue": {"spike":"Promotional burst, flash-sale, or pricing engine over-discount.",
                          "drop": "Payment gateway fault, cart abandonment surge, or SLA breach."},
        "return_rate":   {"spike":"Product quality issue, mis-shipped SKUs, or organised fraud.",
                          "drop": "Return-policy tightening; verify no pipeline suppression."},
        "avg_order_val": {"spike":"B2B order cluster or A/B price variant in production.",
                          "drop": "Coupon stacking, bot micro-orders, or rounding error."},
        "total_orders":  {"spike":"Marketing burst, SEO/viral event, or bot traffic influx.",
                          "drop": "Stockout, checkout regression, or mis-attribution."},
    }
    base = hypotheses.get(met, {}).get(direction, "Confounding factors under investigation.")
    addendum = ""
    if met == "total_revenue" and (rev := kpi.get("total_revenue", 0)):
        addendum = f" Baseline avg ${rev:,.0f} — deviation is {abs(ate)/max(rev,1):.0%} of mean."
    elif met == "return_rate" and (rr := kpi.get("return_rate", 0)):
        addendum = f" Mean return rate {rr:.1%}; breach threshold >15%."
    return f"[{magnitude} {direction.upper()}] {base}{addendum}"



def _build_council_payload(kpi_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> dict:
    """Build a rich natural-language prompt for Gemini so it understands the data."""
    kpi_summary: dict = {}
    for col in ["total_revenue","total_orders","return_rate","avg_order_val"]:
        if col in kpi_df.columns and kpi_df[col].notna().any():
            kpi_summary[col] = float(kpi_df[col].mean())

    top_anom = (anomaly_df.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
                .head(5).to_dict("records") if not anomaly_df.empty else [])

    # Build human-readable anomaly summary for Gemini
    anom_lines = []
    for a in top_anom:
        anom_lines.append(
            f"- {a.get('metric_name','?')} in {a.get('region','?')}: "
            f"observed={a.get('metric_value',0):.2f}, expected={a.get('expected_value',0):.2f}, "
            f"z-score={abs(a.get('z_score',0)):.2f}, severity={a.get('severity','?')}"
        )

    kpi_lines = "\n".join([f"- {k}: {v:.2f}" for k, v in kpi_summary.items()])
    anom_text  = "\n".join(anom_lines) if anom_lines else "No anomalies detected."

    prompt = (
        f"You are analyzing a business KPI dashboard.\n\n"
        f"KPI BASELINE (averages):\n{kpi_lines}\n\n"
        f"DETECTED ANOMALIES:\n{anom_text}\n\n"
        f"Total anomalies: {len(anomaly_df)}\n\n"
        f"Please analyze these anomalies, identify root causes, and recommend actions."
    )
    return {"raw_data": {"prompt": prompt, "kpi_summary": kpi_summary, "anomalies": top_anom}}


def _parse_langgraph_result(res: dict, anomaly_list: list, kpi_summary: dict) -> dict:
    """Map langgraph_agent.py AgentState → council result dict."""
    decision = res.get("final_decision", res.get("decision_output", ""))
    analyst  = res.get("anomaly_report",  res.get("analyst_output",  ""))
    causal   = res.get("business_report", res.get("causal_output",   ""))

    priority = "P4"
    dec_upper = decision.upper()
    if any(k in dec_upper for k in ["REJECT","CRITICAL","IMMEDIATE","P1"]):   priority = "P1"
    elif any(k in dec_upper for k in ["FLAG","URGENT","HIGH","P2"]):           priority = "P2"
    elif any(k in dec_upper for k in ["APPROVE","MONITOR","MEDIUM","P3"]):     priority = "P3"

    # Derive confidence from anomaly z-scores
    max_z = max((abs(a.get("z_score",0)) for a in anomaly_list), default=0)
    conf  = min(0.99, 0.65 + max_z * 0.04)

    return {
        "anomalies":       anomaly_list,
        "kpi_summary":     kpi_summary,
        "analyst_output":  analyst  or "Agent did not return analyst output.",
        "causal_output":   causal   or "Agent did not return causal output.",
        "decision_output": decision or "Agent did not return a decision.",
        "confidence":      conf,
        "final_action":    res.get("final_action", decision[:300] if decision else ""),
        "priority":        priority,
        "completed":       True,
    }


def run_ai_council(kpi_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> dict:
    """
    Primary  : LangGraph + Gemini (langgraph_agent.py → app.invoke)
    Fallback : rule-based 3-agent pipeline (always works, no API needed)
    """
    kpi_summary: dict = {}
    for col in ["total_revenue","total_orders","return_rate","avg_order_val"]:
        if col in kpi_df.columns and kpi_df[col].notna().any():
            kpi_summary[col] = float(kpi_df[col].mean())
    anomaly_list = anomaly_df.to_dict("records") if not anomaly_df.empty else []

    # ── LangGraph + Gemini ────────────────────────────────────────────────────
    if agent_app is not None:
        try:
            payload = _build_council_payload(kpi_df, anomaly_df)
            res     = agent_app.invoke(payload)
            return _parse_langgraph_result(res, anomaly_list, kpi_summary)
        except Exception as _e:
            err_str = str(_e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.sidebar.warning("⚠️ Gemini quota exceeded — using rule-based council.")
            else:
                st.sidebar.warning(f"⚠️ LangGraph error: {err_str[:120]}")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    state: dict = {
        "anomalies":       anomaly_list,
        "kpi_summary":     kpi_summary,
        "analyst_output":  "",
        "causal_output":   "",
        "decision_output": "",
        "confidence":      0.0,
        "final_action":    "",
        "priority":        "P4",
        "completed":       False,
    }
    state = _agent_analyst(state)
    state = _agent_causal(state)
    state = _agent_decision(state)
    return state


def render_ai_council_section(kpi_df: pd.DataFrame, anomaly_df: pd.DataFrame):
    st.markdown('<div class="section-header">🧠 AI Council — Live Decisions</div>', unsafe_allow_html=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    _src_label  = "LangGraph + Gemini" if agent_app is not None else "Rule-Based Fallback"
    _src_color  = "#22c55e" if agent_app is not None else "#f97316"
    _anom_count = len(anomaly_df) if not anomaly_df.empty else 0
    _kpi_rev    = (f"${kpi_df['total_revenue'].mean():,.0f} avg rev"
                   if not kpi_df.empty and "total_revenue" in kpi_df.columns else "no KPI data")
    st.markdown(f"""
    <div class="council-bar">
      <div class="council-agent"><span class="dot"></span> Analyst Agent</div>
      <div class="council-agent"><span class="dot"></span> Causal Agent</div>
      <div class="council-agent"><span class="dot"></span> Decision Agent</div>
      <div style="margin-left:auto;display:flex;gap:1rem;align-items:center;">
        <span style="font-size:.73rem;color:#475569;">{_anom_count} anomalies &nbsp;·&nbsp; {_kpi_rev}</span>
        <span style="font-size:.7rem;font-weight:700;color:{_src_color};background:{_src_color}18;
          border:1px solid {_src_color}44;border-radius:99px;padding:.18rem .65rem;letter-spacing:.08em;">
          ⚙ {_src_label}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 5])
    run_clicked = c1.button("▶ Run Council", key="_council_run_btn",
                            use_container_width=True, type="primary")

    if run_clicked:
        with st.spinner("3 agents analyzing your data…"):
            result = run_ai_council(kpi_df, anomaly_df)
            st.session_state["_council_result"] = result
        st.rerun()  # re-render page so result shows cleanly

    # ── If no result yet, show prompt ─────────────────────────────────────────
    if st.session_state.get("_council_result") is None:
        st.markdown("""
        <div style="background:rgba(10,15,32,.5);border:1px dashed rgba(124,58,237,.25);
          border-radius:12px;padding:2.5rem;text-align:center;margin-top:.5rem;">
          <div style="font-size:2.5rem;margin-bottom:.6rem;">🧠</div>
          <div style="color:#475569;font-size:.9rem;line-height:1.7;">
            Click <b style="color:#a78bfa;">▶ Run Council</b> to start the 3-agent analysis.<br>
            <span style="font-size:.78rem;color:#334155;">
              Analyst → Causal → Decision pipeline will run on your current data.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
        return  # don't try to render result

    # ── Result is ready — render it ───────────────────────────────────────────
    result: dict = st.session_state["_council_result"]

    # ── Priority badge ────────────────────────────────────────────────────────
    _pc     = {"P1":"#ef4444","P2":"#f97316","P3":"#eab308","P4":"#22c55e"}
    p_color = _pc.get(result.get("priority","P4"), "#64748b")
    c2.markdown(f"""
    <div style="display:flex;gap:.8rem;align-items:center;padding:.4rem 0;flex-wrap:wrap;">
      <span style="background:{p_color}1a;border:1px solid {p_color}55;border-radius:99px;
        padding:.22rem .8rem;font-size:.7rem;font-weight:700;color:{p_color};letter-spacing:.1em;">
        {result.get("priority","P4")} PRIORITY
      </span>
      <span style="font-size:.78rem;color:#64748b;">
        Confidence: <b style="color:#a78bfa;">{result.get("confidence",0):.0%}</b>
        &nbsp;·&nbsp; 3 agents &nbsp;·&nbsp; consensus reached
      </span>
    </div>""", unsafe_allow_html=True)

    # ── Agent output tabs ─────────────────────────────────────────────────────
    analyst_out  = result.get("analyst_output","")  or "—"
    causal_out   = result.get("causal_output","")   or "—"
    decision_out = result.get("decision_output","") or "—"
    final_action = result.get("final_action","")    or ""

    ta, tc, td = st.tabs(["🔬 Analyst Agent","🔗 Causal Agent","⚡ Decision Agent"])
    with ta:
        st.markdown(f'<div class="agent-box">{analyst_out}</div>', unsafe_allow_html=True)
    with tc:
        st.markdown(f'<div class="agent-box">{causal_out}</div>',  unsafe_allow_html=True)
    with td:
        st.markdown(f'<div class="agent-box">{decision_out}</div>', unsafe_allow_html=True)
        if final_action:
            st.markdown(f"""
            <div style="margin-top:.8rem;background:linear-gradient(135deg,rgba(124,58,237,.14) 0%,
              rgba(59,130,246,.09) 100%);border:1px solid rgba(124,58,237,.32);border-radius:12px;
              padding:1rem 1.3rem;font-size:.85rem;color:#e2e8f0;">
              <span style="font-size:.62rem;letter-spacing:.16em;color:#7c3aed;font-weight:700;
                display:block;margin-bottom:.4rem;">✦ FINAL ACTION</span>
              {final_action}
            </div>""", unsafe_allow_html=True)

    # ── Full LangGraph JSON (only if real agent ran) ──────────────────────────
    if result.get("completed") and agent_app is not None:
        with st.expander("🔍 Full LangGraph Agent Response", expanded=False):
            st.json({
                "anomaly_report":  analyst_out,
                "business_report": causal_out,
                "final_decision":  decision_out,
            })


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ADVANCED ANALYTICS SECTION
# ══════════════════════════════════════════════════════════════════════════════

_CHART_PALETTE = ["#a78bfa","#60a5fa","#34d399","#f472b6","#fbbf24","#f87171","#38bdf8"]
_DARK_LAYOUT   = dict(plot_bgcolor="rgba(4,7,15,1)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0", font_family="Syne",
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=8,r=8,t=40,b=8),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False))


def _best_cat_col(df, candidates):
    for c in candidates:
        if c in df.columns and 1 < df[c].nunique() <= 20: return c
    return None


def _best_num_cols(df, candidates, n=4):
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:n]


def render_advanced_analytics_section(kpi_df: pd.DataFrame, anomaly_df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        _px = True
    except ImportError:
        _px = False; st.error("pip install plotly")

    pl, pr = st.columns(2)
    with pl:
        cat = _best_cat_col(kpi_df, ["region","category"])
        if cat and not kpi_df.empty and "total_revenue" in kpi_df.columns and _px:
            agg = kpi_df.groupby(cat)["total_revenue"].sum().reset_index()
            agg.columns = [cat,"value"]
            f = px.pie(agg, values="value", names=cat, hole=0.42,
                       title=f"Revenue Share by {cat.replace('_',' ').title()}",
                       color_discrete_sequence=_CHART_PALETTE)
            f.update_traces(textfont_size=11, textfont_color="#e2e8f0")
            f.update_layout(**_DARK_LAYOUT, height=300)
            st.plotly_chart(f, use_container_width=True)
        else:
            render_empty_state("🥧","No data","Upload CSV with region/category.")

    with pr:
        if not anomaly_df.empty and "severity" in anomaly_df.columns and _px:
            sc = anomaly_df["severity"].value_counts().reset_index()
            sc.columns = ["Severity","Count"]
            f2 = px.pie(sc, values="Count", names="Severity", hole=0.42,
                        title="Anomaly Severity Distribution", color="Severity",
                        color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"})
            f2.update_layout(**_DARK_LAYOUT, height=300)
            st.plotly_chart(f2, use_container_width=True)
        else:
            render_empty_state("🥧","No Anomaly Data","Run anomaly detection first.")

    bar_num = _best_num_cols(kpi_df, ["total_revenue","total_orders","avg_order_val","return_rate"], 3)
    bar_cat = _best_cat_col(kpi_df, ["region","category"])
    if bar_cat and bar_num and not kpi_df.empty and _px:
        import plotly.express as px
        agg2 = kpi_df.groupby(bar_cat)[bar_num].mean().reset_index()
        fb = px.bar(agg2, x=bar_cat, y=bar_num[0], color=bar_cat, text_auto=".2s",
                    title=f"Avg {bar_num[0].replace('_',' ').title()} by {bar_cat.replace('_',' ').title()}",
                    color_discrete_sequence=_CHART_PALETTE)
        fb.update_layout(**_DARK_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fb, use_container_width=True)
        if len(bar_num) > 1:
            agg_long = agg2.melt(id_vars=[bar_cat], value_vars=bar_num[1:], var_name="Metric", value_name="Value")
            fb2 = px.bar(agg_long, x=bar_cat, y="Value", color="Metric", barmode="group",
                         title="Secondary KPI Comparison", color_discrete_sequence=_CHART_PALETTE[2:])
            fb2.update_layout(**_DARK_LAYOUT, height=280)
            st.plotly_chart(fb2, use_container_width=True)

    ts_col  = next((c for c in kpi_df.columns if pd.api.types.is_datetime64_any_dtype(kpi_df[c])), None)
    line_num = _best_num_cols(kpi_df, ["total_revenue","total_orders","avg_order_val"], 2)
    if ts_col and line_num and not kpi_df.empty and _px:
        import plotly.express as px
        grp = _best_cat_col(kpi_df, ["region","category"])
        fl  = px.line(kpi_df.groupby([ts_col,grp])[line_num].mean().reset_index() if grp else kpi_df,
                      x=ts_col, y=line_num[0], color=grp if grp else None,
                      title=f"Trend: {line_num[0].replace('_',' ').title()} Over Time",
                      color_discrete_sequence=_CHART_PALETTE)
        fl.update_traces(line_width=1.8)
        fl.update_layout(**_DARK_LAYOUT, height=320)
        st.plotly_chart(fl, use_container_width=True)

    num_all = kpi_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_all) >= 3 and _px:
        import plotly.graph_objects as go
        corr = kpi_df[num_all[:8]].corr()
        fh = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#ef4444"],[.5,"#1e1e3f"],[1,"#22c55e"]], zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont_size=10))
        layout_no_axes = {k:v for k,v in _DARK_LAYOUT.items() if k not in ("xaxis","yaxis")}
        fh.update_layout(**layout_no_axes, title="Correlation Heatmap", height=320)
        with st.expander("🔥 Correlation Heatmap", expanded=False):
            st.plotly_chart(fh, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  DB + DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_engine():
    try:
        from sqlalchemy import create_engine
        from config.settings import config  # type: ignore
        return create_engine(config.db.url, pool_pre_ping=True)
    except Exception:
        return None


def query_db(sql: str, engine=None) -> pd.DataFrame:
    if engine is None: return pd.DataFrame()
    try:
        from sqlalchemy import text
        with engine.connect() as conn: return pd.read_sql(text(sql), conn)
    except Exception as e:
        st.warning(f"DB query failed: {e}"); return pd.DataFrame()


def make_demo_kpis() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hours = pd.date_range(end=datetime.now(timezone.utc), periods=48, freq="h")
    rows = []
    for ts in hours:
        for region in ["North","South","East","West"]:
            base  = 12_000 + rng.normal(0, 800)
            spike = 85_000 if ts.hour == 14 and region == "North" else 0
            rows.append({"snapshot_ts":ts,"region":region,"category":"Electronics",
                         "total_revenue":max(0,base+spike),"total_orders":int(rng.integers(80,200)),
                         "return_rate":float(rng.uniform(.05,.12)),"avg_order_val":float(rng.uniform(60,140))})
    return pd.DataFrame(rows)


def make_demo_anomalies() -> pd.DataFrame:
    return pd.DataFrame([
        {"anomaly_id":1,"metric_name":"total_revenue","metric_value":98450.75,"expected_value":12200.0,
         "z_score":8.94,"severity":"CRITICAL","region":"North","category":"Electronics",
         "detected_at":datetime.now(timezone.utc)-timedelta(hours=1)},
        {"anomaly_id":2,"metric_name":"return_rate","metric_value":0.43,"expected_value":0.07,
         "z_score":5.12,"severity":"HIGH","region":"West","category":"Apparel",
         "detected_at":datetime.now(timezone.utc)-timedelta(hours=3)},
        {"anomaly_id":3,"metric_name":"avg_order_val","metric_value":8.50,"expected_value":95.0,
         "z_score":-4.21,"severity":"HIGH","region":"South","category":"Sports",
         "detected_at":datetime.now(timezone.utc)-timedelta(hours=6)},
    ])


def make_demo_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        {"anomaly_id":1,"priority":"P1","final_action":"Freeze discount promotions in North/Electronics for 4h",
         "status":"PENDING","decided_at":datetime.now(timezone.utc)-timedelta(minutes=45)},
        {"anomaly_id":2,"priority":"P2","final_action":"Review West/Apparel return policy + alert fraud team",
         "status":"IN_PROGRESS","decided_at":datetime.now(timezone.utc)-timedelta(hours=2)},
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 11.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

engine = get_engine()

st.sidebar.image("https://img.icons8.com/fluency/48/artificial-intelligence.png", width=48)
st.sidebar.title("ADI System")
st.sidebar.caption("Autonomous Data Intelligence")

db_status = "🟢 Connected" if engine else "🔴 Demo Mode"
agent_status = "🟢 Gemini LangGraph" if agent_app is not None else ("🟡 Broker only" if broker else "🔴 Rule-Based")
st.sidebar.info(f"Database: {db_status}  \nAI Agent: {agent_status}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Dataset Upload")
st.sidebar.caption("Upload any CSV to override demo data with live intelligence.")

uploaded_file = st.sidebar.file_uploader(
    label="Drop CSV here", type=["csv"],
    help="Arbitrary schema accepted — ADI auto-detects columns.",
    label_visibility="collapsed",
)

# ── Process upload ──────────────────────────────────────────────────────────
if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_csv_file_key") != file_key:
        try:
            raw_df = pd.read_csv(uploaded_file)
            # Universal currency/percent cleaner
            for col in raw_df.select_dtypes(include=["object"]).columns:
                if raw_df[col].astype(str).str.contains(r"\d").any():
                    cleaned   = raw_df[col].astype(str).str.replace(r"[₹$€£,%\s]","",regex=True)
                    converted = pd.to_numeric(cleaned, errors="coerce")
                    if converted.notna().sum() > len(raw_df) * 0.5:
                        raw_df[col] = converted
            proc = auto_process_csv(raw_df)
            st.session_state["_csv_file_name"] = uploaded_file.name   # FIX BUG-9
            st.session_state["_csv_proc"]      = proc
            st.session_state["_csv_kpi_df"]    = _csv_to_kpi_df(proc)
            st.session_state["_csv_anom_df"]   = _csv_to_anomaly_df(proc)
            st.session_state["_csv_insight"]   = generate_csv_insight(proc)
            st.session_state["_csv_file_key"]  = file_key
            st.session_state["_council_result"] = None  # invalidate stale council
            st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Parse error: {e}")

# ── Read from session state ─────────────────────────────────────────────────
csv_proc    = st.session_state.get("_csv_proc")
csv_kpi_df  = st.session_state.get("_csv_kpi_df",  pd.DataFrame())
csv_anom_df = st.session_state.get("_csv_anom_df", pd.DataFrame())
csv_insight = st.session_state.get("_csv_insight", "")

if csv_proc is not None:
    ds_type = csv_proc["dataset_type"].upper()
    st.sidebar.markdown(f"""
    <div style="font-size:.75rem;color:#64748b;margin-top:.3rem;">
      <b style="color:#22c55e;">●</b> {st.session_state.get('_csv_file_name','LIVE CSV')}<br>
      {csv_proc['row_count']:,} rows · {csv_proc['col_count']} cols · <b style="color:#a78bfa;">{ds_type}</b>
    </div>""", unsafe_allow_html=True)
    if st.sidebar.button("✕ Clear dataset", use_container_width=True):
        for k in ["_csv_proc","_csv_kpi_df","_csv_anom_df","_csv_insight","_csv_file_name","_csv_file_key","_council_result"]:
            st.session_state.pop(k, None)
        st.rerun()

st.sidebar.markdown("---")

# ── Stream status ───────────────────────────────────────────────────────────
_stream_running = st.session_state.get("_stream_thread_running", False)
_log_len = len(st.session_state.get("_stream_log", []))
st.sidebar.markdown(f"""
<div style="font-size:.73rem;color:#64748b;margin:.1rem 0 .4rem;">
  Stream: <b style="color:{'#22c55e' if _stream_running else '#94a3b8'};">
  {'🟢 Live' if _stream_running else '⭕ Idle'}</b> &nbsp;·&nbsp; {_log_len} events
</div>""", unsafe_allow_html=True)

# ── Navigation ──────────────────────────────────────────────────────────────
mode = st.sidebar.radio("View", [
    "Overview","Anomalies","Causal Analysis",
    "Agent Decisions","Self-Healing","Pipeline Log",
    "⚡ Live Stream","🧠 AI Council","📊 Advanced Analytics",
], index=st.session_state.get("_active_mode_idx", 0), key="_mode_radio")
st.session_state["_active_mode_idx"] = [
    "Overview","Anomalies","Causal Analysis",
    "Agent Decisions","Self-Healing","Pipeline Log",
    "⚡ Live Stream","🧠 AI Council","📊 Advanced Analytics",
].index(mode)

# ══════════════════════════════════════════════════════════════════════════════
# 12.  DATA LOADING  (FIXED order: DB/demo → CSV override → anomaly method)
# ══════════════════════════════════════════════════════════════════════════════

if engine:
    kpi_df      = query_db("SELECT * FROM v_recent_kpis ORDER BY snapshot_ts DESC LIMIT 500", engine)
    anomaly_df  = query_db("SELECT * FROM anomaly_events ORDER BY detected_at DESC LIMIT 100", engine)
    decision_df = query_db("SELECT * FROM agent_decisions ORDER BY decided_at DESC LIMIT 50", engine)
    pipeline_df = query_db("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 20", engine)
    heal_df     = query_db("SELECT * FROM schema_drift_log ORDER BY detected_at DESC LIMIT 50", engine)
else:
    kpi_df      = make_demo_kpis()
    anomaly_df  = make_demo_anomalies()
    decision_df = make_demo_decisions()
    pipeline_df = pd.DataFrame()
    heal_df     = pd.DataFrame()

# ── CSV override ─────────────────────────────────────────────────────────────
if csv_proc is not None:
    kpi_df = csv_kpi_df if not csv_kpi_df.empty else kpi_df
    if not csv_anom_df.empty:
        anomaly_df = csv_anom_df

# ── Anomaly detection method selector (ONLY when CSV loaded) ─────────────────
if csv_proc is not None and not kpi_df.empty:
    method = st.sidebar.selectbox(
        "Anomaly Detection Method",
        ["Isolation Forest (ML)","Z-Score (Fast)"], index=0,
        key="_anom_method_select"
    )
    z_base = detect_anomalies_zscore(kpi_df)
    if method == "Isolation Forest (ML)":
        ml_df = detect_anomalies_ml(kpi_df)
        if "ml_anomaly" not in ml_df.columns: ml_df["ml_anomaly"] = 1
        anomaly_df = z_base.loc[z_base.index.isin(ml_df.index[ml_df["ml_anomaly"] == -1])]
    else:
        anomaly_df = z_base
    st.session_state["anomaly_df"] = anomaly_df

_dash_title = "ADI System Dashboard"
if csv_proc:
    _dash_title = f"ADI · {csv_proc['dataset_type'].title()} Intelligence"


# ══════════════════════════════════════════════════════════════════════════════
# 13.  VIEWS
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Overview":
    _agent_mode = "LangGraph + Gemini" if agent_app else "Rule-Based Council"
    _data_src   = f"CSV · {csv_proc['row_count']:,} rows" if csv_proc else "Demo Mode"
    _ts         = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="adi-hero fade-in">
      <h1>🤖 ADI System Overview</h1>
      <p class="hero-sub">Last refreshed: {_ts} &nbsp;·&nbsp; {_agent_mode}</p>
      <div class="hero-badges">
        <span class="hero-badge hero-badge-violet">✦ Autonomous Intelligence</span>
        <span class="hero-badge hero-badge-blue">⚡ {_agent_mode}</span>
        <span class="hero-badge {'hero-badge-green' if csv_proc else 'hero-badge-orange'}">
          {'&#9679; ' + _data_src if csv_proc else '&#9675; Demo Mode'}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    if csv_proc:
        render_csv_badge(st.session_state.get("_csv_file_name","data.csv"),
                         csv_proc["row_count"], csv_proc["col_count"])
    if csv_insight:
        render_insight(csv_insight)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rev_total = kpi_df["total_revenue"].sum()
        pct = kpi_df["total_revenue"].pct_change().mean() * 100
        render_kpi_card("💰","Total Revenue (24h)",f"${rev_total:,.0f}",f"{pct:+.1f}%",1 if pct>=0 else -1,delay=1)
    with col2:
        crit = len(anomaly_df[anomaly_df["severity"]=="CRITICAL"]) if "severity" in anomaly_df.columns else 0
        render_kpi_card("🚨","Anomalies Detected",str(len(anomaly_df)),f"{crit} critical",-1 if crit else 0,delay=2)
    with col3:
        render_kpi_card("⚡","Active Decisions",str(len(decision_df)),delay=3)
    with col4:
        rr = kpi_df["return_rate"].mean()*100 if "return_rate" in kpi_df.columns and kpi_df["return_rate"].notna().any() else None
        render_kpi_card("↩️","Avg Return Rate",f"{rr:.1f}%" if rr is not None else "N/A",delay=4)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.subheader("📈 Revenue by Region (48h)")
    try:
        import plotly.express as px
        fig = px.line(kpi_df, x="snapshot_ts", y="total_revenue", color="region",
                      title="Hourly Revenue by Region",
                      color_discrete_sequence=["#a78bfa","#60a5fa","#34d399","#f472b6"])
        fig.update_layout(plot_bgcolor="#080d1a", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=350, margin=dict(l=0,r=0,t=36,b=0),
                          font_family="Syne", legend=dict(bgcolor="rgba(0,0,0,0)"),
                          xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.04)"))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(kpi_df.pivot_table(index="snapshot_ts", columns="region",
                                          values="total_revenue", aggfunc="sum"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔍 Anomaly Severity")
        if not anomaly_df.empty and "severity" in anomaly_df.columns:
            counts = anomaly_df["severity"].value_counts().reset_index()
            counts.columns = ["Severity","Count"]
            try:
                import plotly.express as px
                fig2 = px.pie(counts, values="Count", names="Severity", color="Severity",
                              color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"})
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                                   height=280, font_family="Syne", legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(counts.set_index("Severity"))
        else:
            render_empty_state("🔍","No Anomalies","No anomalies in current dataset.")

    with col_b:
        st.subheader("📋 Latest Anomalies")
        if not anomaly_df.empty:
            cols_show = [c for c in ["metric_name","severity","region","z_score"] if c in anomaly_df.columns]
            st.dataframe(anomaly_df[cols_show].head(5), use_container_width=True)
        else:
            render_empty_state("📋","No Anomalies","Pipeline is clean.")

    if csv_proc:
        with st.expander("🗂️ Explore Raw CSV Data", expanded=False):
            st.markdown(f"""
            <div style="font-size:.8rem;color:#64748b;margin-bottom:.6rem;">
              <b>{csv_proc['col_count']}</b> columns &nbsp;·&nbsp;
              <b>{csv_proc['row_count']:,}</b> rows &nbsp;·&nbsp;
              Dataset type: <b style="color:#a78bfa;">{csv_proc['dataset_type'].upper()}</b>
            </div>""", unsafe_allow_html=True)
            cx, cy = st.columns(2)
            with cx: st.caption("🔢 Numeric"); st.write(csv_proc["numeric_cols"] or ["—"])
            with cy: st.caption("🏷️ Categorical"); st.write(csv_proc["cat_cols"] or ["—"])
            st.dataframe(csv_proc["df"].head(100), use_container_width=True)


elif mode == "Anomalies":
    st.title("🔍 Anomaly Detection")
    if csv_proc:
        render_csv_badge(st.session_state.get("_csv_file_name","data.csv"),
                         csv_proc["row_count"], csv_proc["col_count"])

    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect("Severity",["CRITICAL","HIGH","MEDIUM","LOW"],default=["CRITICAL","HIGH"])
    with col2:
        region_filter = st.multiselect("Region",
                                       anomaly_df["region"].unique().tolist() if not anomaly_df.empty else [],
                                       default=[])
    filtered = anomaly_df.copy()
    if sev_filter and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if region_filter and "region" in filtered.columns:
        filtered = filtered[filtered["region"].isin(region_filter)]

    if filtered.empty:
        render_empty_state("✅","No Anomalies Found","Try broadening severity filters.")
    else:
        st.dataframe(filtered, use_container_width=True, height=400)
        st.subheader("Z-score Distribution")
        if "z_score" in filtered.columns:
            try:
                import plotly.express as px
                fig = px.histogram(filtered, x="z_score", color="severity",
                                   color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d1a",
                                  font_color="#e2e8f0", font_family="Syne")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(filtered["z_score"])


elif mode == "Causal Analysis":
    st.markdown("""
    <div class="adi-hero fade-in" style="padding:1.5rem 2rem 1.2rem;">
      <h1 style="font-size:1.7rem;">🔗 Causal Analysis</h1>
      <p class="hero-sub">ATE (Average Treatment Effect) estimation from live anomaly deltas</p>
    </div>""", unsafe_allow_html=True)

    causal_df = pd.DataFrame()
    if engine:
        causal_df = query_db("SELECT * FROM causal_findings ORDER BY analyzed_at DESC LIMIT 20", engine)

    if causal_df.empty:
        # ── Dynamic: user picks target metric ────────────────────────────────
        if not kpi_df.empty:
            st.markdown('<div class="section-header">📊 Dynamic Causal Impact</div>', unsafe_allow_html=True)
            numeric_cols = kpi_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select Target Metric", numeric_cols, index=0)
                causal_df  = compute_causal_impact(kpi_df, target_col)

    if causal_df.empty:
        # ── Fallback: build from anomaly_df ──────────────────────────────────
        if not anomaly_df.empty:
            kpi_summary_live: dict = {}
            for _col in ["total_revenue","total_orders","return_rate","avg_order_val"]:
                if _col in kpi_df.columns and kpi_df[_col].notna().any():
                    kpi_summary_live[_col] = float(kpi_df[_col].mean())
            _rows = []; _seen: set = set()
            for _a in sorted(anomaly_df.to_dict("records"), key=lambda x: abs(x.get("z_score",0)), reverse=True)[:8]:
                _met = _a.get("metric_name","unknown")
                if _met in _seen: continue
                _seen.add(_met)
                _val = float(_a.get("metric_value",0)); _exp = float(_a.get("expected_value",0))
                _ate = _val - _exp; _z = abs(float(_a.get("z_score",0)))
                _conf = min(0.99, 0.50 + _z*0.07); _pct = (_ate/max(abs(_exp),1e-9))*100
                _rows.append({
                    "anomaly_id": _a.get("anomaly_id","—"), "cause_variable": _met,
                    "effect_variable": "business_outcome", "ate": round(_ate,4),
                    "ate_pct": round(_pct,1), "confidence": round(_conf,3),
                    "z_score": round(_z,2), "severity": _a.get("severity","MEDIUM"),
                    "region": _a.get("region","—"), "method": "Z-score + ATE (live)",
                    "explanation": _causal_narrative(_met, _ate, _z, kpi_summary_live),
                })
            causal_df = pd.DataFrame(_rows)

    if causal_df.empty:
        render_empty_state("🔗","No Causal Findings","Upload a CSV or connect a DB.")
    else:
        # ── ATE bar chart ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 ATE Impact Overview</div>', unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            _sev_c = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
            _bcols  = [_sev_c.get(s,"#a78bfa") for s in causal_df.get("severity", pd.Series(["MEDIUM"]*len(causal_df)))]
            fig_ate = go.Figure(go.Bar(
                x=causal_df["cause_variable"], y=causal_df["ate"],
                marker_color=_bcols,
                text=[f"{v:+,.2f}" for v in causal_df["ate"]],
                textposition="outside", textfont=dict(size=11,color="#e2e8f0")))
            fig_ate.update_layout(plot_bgcolor="rgba(4,7,15,1)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e2e8f0", font_family="Syne", height=320,
                                  xaxis_title="Metric", yaxis_title="ATE (Observed − Expected)",
                                  margin=dict(l=8,r=8,t=20,b=8))
            st.plotly_chart(fig_ate, use_container_width=True)
        except ImportError:
            st.bar_chart(causal_df.set_index("cause_variable")["ate"])

        # ── Causal detail cards ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Causal Chain Details</div>', unsafe_allow_html=True)
        for _, row in causal_df.iterrows():
            _sev    = row.get("severity","MEDIUM")
            _sc     = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}.get(_sev,"#a78bfa")
            _ate_v  = row.get("ate",0); _conf = row.get("confidence",0)
            _pct    = row.get("ate_pct", (row.get("ate",0)/max(abs(row.get("confidence",1)),1e-9))*100)
            _expl   = row.get("explanation","—")
            st.markdown(f"""
            <div class="causal-card">
              <div class="cc-metric">{row.get("cause_variable","?")} ──▶ {row.get("effect_variable","?")}</div>
              <div style="display:flex;align-items:baseline;gap:1.2rem;flex-wrap:wrap;">
                <div>
                  <div class="cc-ate">{_ate_v:+,.4f}</div>
                  <div style="font-size:.68rem;color:#475569;margin-top:.15rem;">ATE &nbsp;·&nbsp; {_pct:+.1f}% from mean</div>
                </div>
                <div>
                  <div class="cc-conf">&#x25CF; {_conf:.0%} confidence</div>
                  <div style="font-size:.68rem;color:#475569;margin-top:.15rem;">
                    z={row.get("z_score","?")} &nbsp;·&nbsp; {row.get("region","—")}
                  </div>
                </div>
                <span style="margin-left:auto;background:{_sc}18;border:1px solid {_sc}44;
                  border-radius:99px;padding:.18rem .65rem;font-size:.67rem;font-weight:700;
                  color:{_sc};letter-spacing:.08em;">{_sev}</span>
              </div>
              <div class="cc-hyp">{_expl}</div>
            </div>""", unsafe_allow_html=True)


elif mode == "Agent Decisions":
    st.title("🤝 Multi-Agent Reasoning")
    st.info("Three LLM-powered agents debate the anomaly and agree on a recommended action.")
    if decision_df.empty:
        render_empty_state("🤝","No Decisions Yet","Agent decisions appear here once anomalies are processed.")
    else:
        for _, row in decision_df.iterrows():
            pcolor = {"P0":"🔴","P1":"🟠","P2":"🟡","P3":"🟢"}.get(row.get("priority","P2"),"⚪")
            with st.expander(f"{pcolor} [{row.get('priority','?')}] Anomaly #{row.get('anomaly_id','?')}"
                             f" — {str(row.get('final_action',''))[:80]}…"):
                st.markdown(f"**Priority:** `{row.get('priority')}`  \n**Status:** `{row.get('status','PENDING')}`")
                t1,t2,t3 = st.tabs(["🔬 Analyst","🔗 Causal","⚡ Decision"])
                with t1: st.markdown(f"<div class='agent-box'>{row.get('analyst_output','N/A')}</div>", unsafe_allow_html=True)
                with t2: st.markdown(f"<div class='agent-box'>{row.get('causal_output','N/A')}</div>",  unsafe_allow_html=True)
                with t3: st.markdown(f"<div class='agent-box'>{row.get('decision_output','N/A')}</div>",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🗄️ Actionable AI Live Database Logs</div>', unsafe_allow_html=True)
    db_path = "autonomous_system.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            saved_df = pd.read_sql_query("SELECT * FROM ai_audit_log ORDER BY id DESC", conn)
            conn.close()
            if not saved_df.empty:
                st.dataframe(saved_df, use_container_width=True)
            else:
                st.warning("Database exists but no decisions saved yet. Run the Live Stream first!")
        except Exception as e:
            st.error(f"Error reading database: {e}")
    else:
        st.info("Database file ('autonomous_system.db') not found. It will be created once the backend runs.")


elif mode == "Self-Healing":
    st.title("🔧 Self-Healing Pipeline")
    if not heal_df.empty:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Drifts", len(heal_df))
        c2.metric("Auto-Healed",  int(heal_df["auto_healed"].sum()) if "auto_healed" in heal_df.columns else 0)
        c3.metric("Pending Review", int((~heal_df["auto_healed"]).sum()) if "auto_healed" in heal_df.columns else 0)
        st.dataframe(heal_df, use_container_width=True)
    else:
        st.success("✅ No schema drift detected. Pipeline is healthy.")
        st.json({"status":"HEALTHY","last_check":datetime.now(timezone.utc).isoformat(),
                 "tables_monitored":["raw_orders","raw_events","kpi_snapshots"]})


elif mode == "Pipeline Log":
    st.title("📋 Pipeline Run Log")
    if not pipeline_df.empty:
        st.dataframe(pipeline_df, use_container_width=True)
    else:
        render_empty_state("📋","No Pipeline Runs","Run `python main.py --mode run` to see logs here.")


elif mode == "⚡ Live Stream":
    # ══════════════════════════════════════════════════════════════════════════
    # LIVE STREAM — real-time event feed, Gemini analyzes each new event auto
    # আলাদা: AI Council = সব anomaly একসাথে macro verdict
    #         Live Stream = একটা একটা event real-time → auto Gemini analysis
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="adi-hero fade-in" style="padding:1.4rem 2rem 1.2rem;">
      <h1 style="font-size:1.8rem;">⚡ Live Event Stream</h1>
      <p class="hero-sub">
        New anomaly events arrive every few seconds →
        Gemini agent analyzes each one → decisions logged automatically
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Stream controls ───────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1,1,1,2])
    stream_on  = ctrl1.toggle("▶ Stream", value=st.session_state.get("_ls_stream_on", False),
                               key="_ls_stream_on_toggle")
    auto_anal  = ctrl2.toggle("🤖 Auto-analyze", value=st.session_state.get("_ls_auto_anal", True),
                               key="_ls_auto_anal_toggle")
    if ctrl3.button("🗑 Clear All", key="_ls_clear"):
        for k in ["_ls_event_log","_ls_analysis_log","_ls_event_counter"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.session_state["_ls_stream_on"]  = stream_on
    st.session_state["_ls_auto_anal"]  = auto_anal

    # Count & status
    ev_log   = st.session_state.get("_ls_event_log",   [])
    anal_log = st.session_state.get("_ls_analysis_log",[])
    ctrl4.markdown(f"""
    <div style="font-size:.78rem;color:#64748b;padding:.4rem 0;">
      {'🟢 Streaming' if stream_on else '⏸ Paused'} &nbsp;·&nbsp;
      {len(ev_log)} events &nbsp;·&nbsp;
      {len(anal_log)} analyzed &nbsp;·&nbsp;
      🤖 {'Auto-analyze ON' if auto_anal else 'Manual mode'}
    </div>""", unsafe_allow_html=True)

    # ── Generate new event from real anomaly data ─────────────────────────────
    if stream_on:
        counter = st.session_state.get("_ls_event_counter", 0)
        # Pick next anomaly from anomaly_df (cycle through)
        if not anomaly_df.empty:
            row = anomaly_df.iloc[counter % len(anomaly_df)]
            new_ev = {
                "event_id":      counter + 1,
                "ts":            datetime.now().strftime("%H:%M:%S"),
                "metric":        row.get("metric_name","?"),
                "region":        row.get("region","?"),
                "severity":      row.get("severity","MEDIUM"),
                "observed":      float(row.get("metric_value",0)),
                "expected":      float(row.get("expected_value",0)),
                "z_score":       float(row.get("z_score",0)),
                "analyzed":      False,
                "analysis":      None,
            }
        else:
            # Synthetic event if no anomaly data
            _metrics   = ["total_revenue","return_rate","avg_order_val","total_orders"]
            _regions   = ["North","South","East","West"]
            _sevs      = ["HIGH","CRITICAL","MEDIUM","HIGH"]
            _vals      = [98450, 0.43, 8.5, 1200]
            _exps      = [12200, 0.07, 95.0, 180]
            _zs        = [8.9,   5.1,  -4.2, 6.7]
            i = counter % 4
            new_ev = {
                "event_id": counter+1, "ts": datetime.now().strftime("%H:%M:%S"),
                "metric": _metrics[i], "region": _regions[i],
                "severity": _sevs[i],  "observed": _vals[i],
                "expected": _exps[i],  "z_score": _zs[i],
                "analyzed": False, "analysis": None,
            }

        ev_log = [new_ev] + ev_log
        st.session_state["_ls_event_log"]     = ev_log[:30]
        st.session_state["_ls_event_counter"] = counter + 1

        # Auto-analyze with Gemini if enabled
        if auto_anal and agent_app is not None:
            try:
                prompt_data = {
                    "raw_data": (
                        f"SINGLE EVENT ALERT:\n"
                        f"Metric  : {new_ev['metric']}\n"
                        f"Region  : {new_ev['region']}\n"
                        f"Severity: {new_ev['severity']}\n"
                        f"Observed: {new_ev['observed']:.2f}  (Expected: {new_ev['expected']:.2f})\n"
                        f"Z-score : {new_ev['z_score']:.2f}σ\n\n"
                        f"Provide: (1) what this anomaly means, "
                        f"(2) likely root cause, (3) immediate action in 3 bullet points."
                    )
                }
                res = agent_app.invoke(prompt_data)
                ev_log[0]["analyzed"] = True
                ev_log[0]["analysis"] = {
                    "anomaly_report":  res.get("anomaly_report",""),
                    "business_report": res.get("business_report",""),
                    "final_decision":  res.get("final_decision",""),
                }
                anal_log = [ev_log[0]] + anal_log
                st.session_state["_ls_event_log"]     = ev_log
                st.session_state["_ls_analysis_log"]  = anal_log[:20]
            except Exception as _e:
                ev_log[0]["analysis"] = {"final_decision": f"Analysis error: {str(_e)[:80]}"}

    # ── Two panel layout ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown('<div class="section-header">📡 Incoming Event Feed</div>',
                    unsafe_allow_html=True)
        if not ev_log:
            st.markdown("""
            <div style="background:rgba(10,15,32,.5);border:1px dashed rgba(124,58,237,.2);
              border-radius:10px;padding:1.5rem;text-align:center;color:#475569;font-size:.85rem;">
              Toggle <b style="color:#a78bfa;">▶ Stream</b> above to start receiving events.
            </div>""", unsafe_allow_html=True)
        else:
            _sev_col  = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
            _sev_icon = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}
            for ev in ev_log[:12]:
                sev   = ev.get("severity","MEDIUM")
                color = _sev_col.get(sev,"#64748b")
                icon  = _sev_icon.get(sev,"⚪")
                analyzed_dot = '<span style="color:#22c55e;">✓</span>' if ev.get("analyzed") else '<span style="color:#334155;">○</span>'
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.7rem;padding:.6rem .8rem;
                  border-left:3px solid {color};border-radius:8px;
                  background:rgba(10,14,28,.7);margin-bottom:4px;
                  font-family:'DM Mono',monospace;font-size:.76rem;">
                  <span>{icon}</span>
                  <div style="flex:1;">
                    <span style="color:{color};font-weight:700;">
                      {str(ev.get('metric','')).replace('_',' ').upper()}
                    </span>
                    <span style="color:#475569;"> · {ev.get('region','')} · z={ev.get('z_score',0):.1f}σ</span>
                    <div style="color:#334155;font-size:.68rem;">
                      {ev.get('observed',0):,.1f} vs {ev.get('expected',0):,.1f} exp
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <div style="font-size:.68rem;color:#334155;">#{ev.get('event_id','')} {ev.get('ts','')}</div>
                    <div>{analyzed_dot}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Manual analyze button for top unanalyzed event
        unanalyzed = [e for e in ev_log if not e.get("analyzed")]
        if unanalyzed:
            top_ev = unanalyzed[0]
            if st.button(f"🤖 Analyze Event #{top_ev['event_id']} manually",
                         use_container_width=True, key="_ls_manual_analyze"):
                with st.spinner(f"Gemini analyzing {top_ev['metric']}…"):
                    try:
                        prompt_data = {
                            "raw_data": (
                                f"ANOMALY EVENT:\n"
                                f"Metric={top_ev['metric']}, Region={top_ev['region']}, "
                                f"Severity={top_ev['severity']}, Observed={top_ev['observed']:.2f}, "
                                f"Expected={top_ev['expected']:.2f}, Z-score={top_ev['z_score']:.2f}σ\n"
                                f"What happened, why, and what should we do?"
                            )
                        }
                        if agent_app:
                            res = agent_app.invoke(prompt_data)
                        else:
                            # Rule-based fallback
                            single = pd.DataFrame([{
                                "metric_name": top_ev["metric"], "metric_value": top_ev["observed"],
                                "expected_value": top_ev["expected"], "z_score": top_ev["z_score"],
                                "severity": top_ev["severity"], "region": top_ev["region"],
                            }])
                            rb = run_ai_council(kpi_df, single)
                            res = {"anomaly_report": rb["analyst_output"],
                                   "business_report": rb["causal_output"],
                                   "final_decision": rb["decision_output"]}

                        for e in ev_log:
                            if e["event_id"] == top_ev["event_id"]:
                                e["analyzed"] = True
                                e["analysis"] = res
                                break
                        st.session_state["_ls_event_log"] = ev_log
                        anal_log = [e for e in ev_log if e.get("analyzed")]
                        st.session_state["_ls_analysis_log"] = anal_log[:20]
                    except Exception as _ex:
                        st.error(f"Error: {_ex}")
                st.rerun()

    with right:
        st.markdown('<div class="section-header">🧠 Gemini Analysis Results</div>',
                    unsafe_allow_html=True)
        analyzed = [e for e in ev_log if e.get("analyzed") and e.get("analysis")]
        if not analyzed:
            st.markdown("""
            <div style="background:rgba(10,15,32,.5);border:1px dashed rgba(124,58,237,.2);
              border-radius:10px;padding:1.5rem;text-align:center;color:#475569;font-size:.85rem;">
              <div style="font-size:1.8rem;margin-bottom:.4rem;">🤖</div>
              Enable <b style="color:#a78bfa;">Auto-analyze</b> or click analyze button.<br>
              Gemini will explain each event in detail.
            </div>""", unsafe_allow_html=True)
        else:
            _sev_col = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
            for ev in analyzed[:5]:
                an    = ev["analysis"]
                sev   = ev.get("severity","MEDIUM")
                color = _sev_col.get(sev,"#64748b")
                met   = str(ev.get("metric","")).replace("_"," ").upper()
                with st.expander(
                    f"#{ev['event_id']} · {met} · {ev.get('region','')} · {ev.get('ts','')}",
                    expanded=(ev == analyzed[0])
                ):
                    t1, t2, t3 = st.tabs(["🔬 Anomaly Report","📊 Business Report","⚡ Decision"])
                    with t1:
                        st.markdown(
                            f'<div class="agent-box" style="font-size:.76rem;">'
                            f'{an.get("anomaly_report","") or "—"}</div>',
                            unsafe_allow_html=True)
                    with t2:
                        st.markdown(
                            f'<div class="agent-box" style="font-size:.76rem;">'
                            f'{an.get("business_report","") or "—"}</div>',
                            unsafe_allow_html=True)
                    with t3:
                        dec = an.get("final_decision","") or "—"
                        _dc = "#ef4444" if any(k in dec.upper() for k in ["REJECT","CRITICAL","FLAG"]) else "#22c55e"
                        st.markdown(f"""
                        <div style="border-left:3px solid {_dc};padding:.8rem 1rem;
                          background:rgba(10,14,28,.8);border-radius:8px;
                          font-family:'DM Mono',monospace;font-size:.78rem;color:#e2e8f0;">
                          {dec}
                        </div>""", unsafe_allow_html=True)

    # Auto-refresh while streaming
    if stream_on:
        time.sleep(3)
        st.rerun()
elif mode == "🧠 AI Council":
    # ══════════════════════════════════════════════════════════════════════════
    # AI COUNCIL — সব anomaly একসাথে → 3 agent পুরো debate করে → final verdict
    # Live Stream থেকে আলাদা: এখানে macro-level full dataset analysis
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="adi-hero fade-in">
      <h1>🧠 AI Council — Full Dataset Verdict</h1>
      <p class="hero-sub">
        All {len(anomaly_df)} anomalies analyzed together · 3 specialized agents debate · consensus verdict
      </p>
      <div class="hero-badges">
        <span class="hero-badge hero-badge-violet">✦ Macro Analysis</span>
        <span class="hero-badge {'hero-badge-green' if agent_app else 'hero-badge-orange'}">
          ⚙ {'LangGraph + Gemini' if agent_app else 'Rule-Based Council'}
        </span>
        <span class="hero-badge hero-badge-blue">
          📊 {len(anomaly_df)} anomalies · {len(kpi_df)} KPI records
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Data being analyzed ───────────────────────────────────────────────────
    if not anomaly_df.empty:
        _sev_counts = anomaly_df.get("severity", pd.Series()).value_counts().to_dict() if "severity" in anomaly_df.columns else {}
        _sev_line = "  ".join([
            f'<span style="color:{"#ef4444" if s=="CRITICAL" else "#f97316" if s=="HIGH" else "#eab308"};font-weight:700;">'
            f'{s}: {c}</span>'
            for s, c in _sev_counts.items()
        ])
        _top5 = anomaly_df.sort_values("z_score", key=lambda x: x.abs(), ascending=False).head(5)
        _metrics_in = ", ".join(_top5["metric_name"].unique()) if "metric_name" in _top5.columns else "N/A"

        st.markdown(f"""
        <div style="background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.18);
          border-radius:12px;padding:.9rem 1.2rem;margin-bottom:1rem;
          font-family:'DM Mono',monospace;font-size:.8rem;color:#94a3b8;">
          <span style="color:#7c3aed;font-weight:700;font-size:.65rem;letter-spacing:.12em;">
            COUNCIL INPUT DATA
          </span><br>
          <span>Metrics: <b style="color:#a78bfa;">{_metrics_in}</b></span><br>
          <span>Severity breakdown: {_sev_line}</span><br>
          <span>Top z-score: <b style="color:#f1f5f9;">
            {abs(_top5.iloc[0].get("z_score",0)) if len(_top5) else 0:.2f}σ
          </b> in <b style="color:#60a5fa;">
            {_top5.iloc[0].get("region","?") if len(_top5) else "N/A"}
          </b></span>
        </div>""", unsafe_allow_html=True)

    # ── Council status bar ────────────────────────────────────────────────────
    _src_color = "#22c55e" if agent_app else "#f97316"
    _src_label = "LangGraph + Gemini 2.5 Flash" if agent_app else "Rule-Based (Gemini unavailable)"
    st.markdown(f"""
    <div class="council-bar">
      <div class="council-agent"><span class="dot"></span> 🔬 Anomaly Expert</div>
      <div class="council-agent"><span class="dot"></span> 📊 Business Strategist</div>
      <div class="council-agent"><span class="dot"></span> ⚖️ Executive Reviewer</div>
      <div style="margin-left:auto;">
        <span style="font-size:.7rem;font-weight:700;color:{_src_color};
          background:{_src_color}18;border:1px solid {_src_color}44;
          border-radius:99px;padding:.18rem .65rem;">⚙ {_src_label}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 3])
    run_clicked  = c1.button("▶ Run Full Council", key="_council_run_btn",
                              use_container_width=True, type="primary")
    clear_clicked = c2.button("↺ Clear Result", key="_council_clear_btn",
                               use_container_width=True)

    if clear_clicked:
        st.session_state.pop("_council_result", None)
        st.rerun()

    if run_clicked:
        with st.spinner("3 agents analyzing all anomalies together…"):
            result = run_ai_council(kpi_df, anomaly_df)
            st.session_state["_council_result"] = result
        st.rerun()

    # ── Prompt if not yet run ─────────────────────────────────────────────────
    if st.session_state.get("_council_result") is None:
        st.markdown("""
        <div style="background:rgba(10,15,32,.5);border:1px dashed rgba(124,58,237,.25);
          border-radius:14px;padding:3rem;text-align:center;margin-top:.5rem;">
          <div style="font-size:3rem;margin-bottom:.8rem;">🧠</div>
          <div style="color:#475569;font-size:.95rem;line-height:1.8;">
            Click <b style="color:#a78bfa;">▶ Run Full Council</b> above.<br>
            <span style="color:#334155;font-size:.8rem;">
              All 3 agents will debate your entire anomaly dataset<br>
              and reach a consensus verdict with priority action.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        result = st.session_state["_council_result"]
        _pc    = {"P1":"#ef4444","P2":"#f97316","P3":"#eab308","P4":"#22c55e"}
        p      = result.get("priority","P4")
        pc     = _pc.get(p,"#64748b")
        conf   = result.get("confidence", 0)

        # ── Verdict banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{pc}12 0%,rgba(10,15,32,.8) 100%);
          border:1.5px solid {pc}44;border-radius:14px;padding:1.2rem 1.6rem;margin:.8rem 0 1rem;">
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
            <span style="font-size:1.6rem;">
              {'🔴' if p=='P1' else '🟠' if p=='P2' else '🟡' if p=='P3' else '🟢'}
            </span>
            <div>
              <div style="font-size:1.1rem;font-weight:800;color:{pc};letter-spacing:.04em;">
                {p} PRIORITY — COUNCIL VERDICT
              </div>
              <div style="font-size:.76rem;color:#64748b;margin-top:.2rem;">
                Confidence: <b style="color:#a78bfa;">{conf:.0%}</b>
                &nbsp;·&nbsp; 3 agents &nbsp;·&nbsp; full dataset analysis
                &nbsp;·&nbsp; {len(anomaly_df)} anomalies reviewed
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 3 agent tabs ──────────────────────────────────────────────────────
        ta, tc, td = st.tabs([
            "🔬 Anomaly Expert (Agent 1)",
            "📊 Business Strategist (Agent 2)",
            "⚖️ Executive Reviewer (Agent 3)"
        ])
        with ta:
            st.markdown(
                f'<div class="agent-box">{result.get("analyst_output","") or "—"}</div>',
                unsafe_allow_html=True)
        with tc:
            st.markdown(
                f'<div class="agent-box">{result.get("causal_output","") or "—"}</div>',
                unsafe_allow_html=True)
        with td:
            st.markdown(
                f'<div class="agent-box">{result.get("decision_output","") or "—"}</div>',
                unsafe_allow_html=True)
            fa = result.get("final_action","")
            if fa:
                st.markdown(f"""
                <div style="margin-top:.9rem;background:linear-gradient(135deg,
                  rgba(124,58,237,.15) 0%,rgba(59,130,246,.10) 100%);
                  border:1px solid rgba(124,58,237,.35);border-radius:12px;
                  padding:1.1rem 1.4rem;color:#e2e8f0;font-size:.87rem;line-height:1.7;">
                  <div style="font-size:.62rem;letter-spacing:.16em;color:#7c3aed;
                    font-weight:700;margin-bottom:.5rem;">✦ FINAL ACTION PLAN</div>
                  {fa}
                </div>""", unsafe_allow_html=True)


elif mode == "📊 Advanced Analytics":
    render_advanced_analytics_section(kpi_df, anomaly_df)