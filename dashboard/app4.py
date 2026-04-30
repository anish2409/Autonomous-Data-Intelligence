"""
dashboard/app.py
Streamlit dashboard for the Autonomous Data Intelligence System.
Run with: streamlit run dashboard/app.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADI System Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e; border-radius: 12px; padding: 1.2rem;
    border-left: 4px solid #7c3aed; margin-bottom: 0.5rem;
  }
  .severity-CRITICAL { color: #ef4444; font-weight: bold; }
  .severity-HIGH     { color: #f97316; font-weight: bold; }
  .severity-MEDIUM   { color: #eab308; font-weight: bold; }
  .severity-LOW      { color: #22c55e; font-weight: bold; }
  .agent-box {
    background: #0f172a; border-radius: 8px; padding: 1rem;
    font-family: monospace; font-size: 0.85rem; margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── DB connection (optional) ─────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    try:
        from sqlalchemy import create_engine
        from config.settings import config
        return create_engine(config.db.url, pool_pre_ping=True)
    except Exception:
        return None


def query_db(sql: str, engine=None) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as e:
        st.warning(f"DB query failed: {e}")
        return pd.DataFrame()


# ── Synthetic demo data ───────────────────────────────────────────────────────
def make_demo_kpis() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hours = pd.date_range(end=datetime.now(timezone.utc), periods=48, freq="h")
    rows  = []
    for ts in hours:
        for region in ["North", "South", "East", "West"]:
            base    = 12_000 + rng.normal(0, 800)
            spike   = 85_000 if ts.hour == 14 and region == "North" else 0
            rows.append({
                "snapshot_ts":   ts,
                "region":        region,
                "category":      "Electronics",
                "total_revenue": max(0, base + spike),
                "total_orders":  int(rng.integers(80, 200)),
                "return_rate":   float(rng.uniform(0.05, 0.12)),
                "avg_order_val": float(rng.uniform(60, 140)),
            })
    return pd.DataFrame(rows)


def make_demo_anomalies() -> pd.DataFrame:
    return pd.DataFrame([
        {"anomaly_id": 1, "metric_name": "total_revenue",
         "metric_value": 98450.75, "expected_value": 12200.0,
         "z_score": 8.94, "severity": "CRITICAL",
         "region": "North", "category": "Electronics",
         "detected_at": datetime.now(timezone.utc) - timedelta(hours=1)},
        {"anomaly_id": 2, "metric_name": "return_rate",
         "metric_value": 0.43, "expected_value": 0.07,
         "z_score": 5.12, "severity": "HIGH",
         "region": "West", "category": "Apparel",
         "detected_at": datetime.now(timezone.utc) - timedelta(hours=3)},
        {"anomaly_id": 3, "metric_name": "avg_order_val",
         "metric_value": 8.50, "expected_value": 95.0,
         "z_score": -4.21, "severity": "HIGH",
         "region": "South", "category": "Sports",
         "detected_at": datetime.now(timezone.utc) - timedelta(hours=6)},
    ])


def make_demo_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        {"anomaly_id": 1, "priority": "P1",
         "final_action": "Freeze discount promotions in North/Electronics for 4h",
         "status": "PENDING",
         "decided_at": datetime.now(timezone.utc) - timedelta(minutes=45)},
        {"anomaly_id": 2, "priority": "P2",
         "final_action": "Review West/Apparel return policy + alert fraud team",
         "status": "IN_PROGRESS",
         "decided_at": datetime.now(timezone.utc) - timedelta(hours=2)},
    ])


# ── Sidebar ────────────────────────────────────────────────────────────────────
engine = get_engine()

st.sidebar.image("https://img.icons8.com/fluency/48/artificial-intelligence.png", width=48)
st.sidebar.title("ADI System")
st.sidebar.caption("Autonomous Data Intelligence")

db_status = "🟢 Connected" if engine else "🔴 Demo Mode"
st.sidebar.info(f"Database: {db_status}")

mode = st.sidebar.radio("View", ["Overview", "Anomalies", "Causal Analysis",
                                  "Agent Decisions", "Self-Healing", "Pipeline Log"])

# ── Data loading ──────────────────────────────────────────────────────────────
if engine:
    kpi_df       = query_db("SELECT * FROM v_recent_kpis ORDER BY snapshot_ts DESC LIMIT 500", engine)
    anomaly_df   = query_db("SELECT * FROM anomaly_events ORDER BY detected_at DESC LIMIT 100", engine)
    decision_df  = query_db("SELECT * FROM agent_decisions ORDER BY decided_at DESC LIMIT 50", engine)
    pipeline_df  = query_db("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 20", engine)
    heal_df      = query_db("SELECT * FROM schema_drift_log ORDER BY detected_at DESC LIMIT 50", engine)
else:
    kpi_df      = make_demo_kpis()
    anomaly_df  = make_demo_anomalies()
    decision_df = make_demo_decisions()
    pipeline_df = pd.DataFrame()
    heal_df     = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "Overview":
    st.title("🤖 Autonomous Data Intelligence System")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue (24h)",
                f"${kpi_df['total_revenue'].sum():,.0f}",
                delta=f"+{kpi_df['total_revenue'].pct_change().mean()*100:.1f}%")
    col2.metric("Anomalies Detected", len(anomaly_df),
                delta=f"{len(anomaly_df[anomaly_df['severity']=='CRITICAL'])} critical")
    col3.metric("Active Decisions", len(decision_df))
    col4.metric("Avg Return Rate",
                f"{kpi_df['return_rate'].mean()*100:.1f}%" if 'return_rate' in kpi_df else "N/A")

    st.markdown("---")

    # Revenue time series
    st.subheader("📈 Revenue by Region (48h)")
    try:
        import plotly.express as px
        fig = px.line(
            kpi_df, x="snapshot_ts", y="total_revenue", color="region",
            title="Hourly Revenue by Region",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(
            kpi_df.pivot_table(index="snapshot_ts", columns="region",
                               values="total_revenue", aggfunc="sum")
        )

    # Anomaly severity breakdown
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔍 Anomaly Severity")
        if not anomaly_df.empty and "severity" in anomaly_df.columns:
            counts = anomaly_df["severity"].value_counts().reset_index()
            counts.columns = ["Severity", "Count"]
            try:
                import plotly.express as px
                fig2 = px.pie(counts, values="Count", names="Severity",
                              color="Severity",
                              color_discrete_map={
                                  "CRITICAL": "#ef4444", "HIGH": "#f97316",
                                  "MEDIUM": "#eab308", "LOW": "#22c55e"
                              })
                fig2.update_layout(paper_bgcolor="#0f172a", font_color="#e2e8f0", height=280)
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(counts.set_index("Severity"))

    with col_b:
        st.subheader("📋 Latest Anomalies")
        disp = anomaly_df[["metric_name", "severity", "region", "z_score"]].head(5)
        st.dataframe(disp, use_container_width=True)


elif mode == "Anomalies":
    st.title("🔍 Anomaly Detection")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"],
                                    default=["CRITICAL","HIGH"])
    with col2:
        region_filter = st.multiselect("Region",
                                       anomaly_df["region"].unique().tolist()
                                       if not anomaly_df.empty else [],
                                       default=[])

    filtered = anomaly_df.copy()
    if sev_filter:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]

    st.dataframe(filtered, use_container_width=True, height=400)

    if not filtered.empty:
        st.subheader("Z-score Distribution")
        try:
            import plotly.express as px
            fig = px.histogram(filtered, x="z_score", color="severity",
                               color_discrete_map={
                                   "CRITICAL": "#ef4444", "HIGH": "#f97316",
                                   "MEDIUM": "#eab308", "LOW": "#22c55e"
                               })
            fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(filtered["z_score"])


elif mode == "Causal Analysis":
    st.title("🔗 Causal Analysis")
    st.info("Causal findings explain **why** an anomaly occurred using DoWhy ATE estimation.")

    if engine:
        causal_df = query_db(
            "SELECT * FROM causal_findings ORDER BY analyzed_at DESC LIMIT 20", engine
        )
    else:
        causal_df = pd.DataFrame([{
            "finding_id": 1, "anomaly_id": 1,
            "cause_variable": "avg_discount", "effect_variable": "total_revenue",
            "ate": 124.57, "confidence": 0.87, "method": "dowhy_backdoor_lr",
            "explanation": "avg_discount caused total_revenue anomaly in North/Electronics "
                           "with ATE=124.5700 and confidence=87.00%.",
        }])

    for _, row in causal_df.iterrows():
        with st.expander(
            f"Anomaly #{row['anomaly_id']} | "
            f"{row['cause_variable']} → {row['effect_variable']} "
            f"| Confidence: {row['confidence']:.0%}"
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("ATE", f"{row['ate']:+.4f}")
            col2.metric("Confidence", f"{row['confidence']:.0%}")
            col3.metric("Method", row["method"])
            st.markdown(f"**Explanation:** {row['explanation']}")


elif mode == "Agent Decisions":
    st.title("🤝 Multi-Agent Reasoning")
    st.info("Three LLM-powered agents debate the anomaly and agree on a recommended action.")

    for _, row in decision_df.iterrows():
        priority_color = {
            "P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"
        }.get(row.get("priority", "P2"), "⚪")

        with st.expander(
            f"{priority_color} [{row.get('priority','?')}] Anomaly #{row.get('anomaly_id','?')}"
            f" — {row.get('final_action','')[:80]}…"
        ):
            st.markdown(f"**Priority:** `{row.get('priority')}`")
            st.markdown(f"**Status:** `{row.get('status','PENDING')}`")

            tab1, tab2, tab3 = st.tabs(["🔬 Analyst", "🔗 Causal Agent", "⚡ Decision"])
            with tab1:
                st.markdown(f"<div class='agent-box'>{row.get('analyst_output','N/A')}</div>",
                            unsafe_allow_html=True)
            with tab2:
                st.markdown(f"<div class='agent-box'>{row.get('causal_output','N/A')}</div>",
                            unsafe_allow_html=True)
            with tab3:
                st.markdown(f"<div class='agent-box'>{row.get('decision_output','N/A')}</div>",
                            unsafe_allow_html=True)


elif mode == "Self-Healing":
    st.title("🔧 Self-Healing Pipeline")

    if not heal_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Drifts", len(heal_df))
        col2.metric("Auto-Healed",
                    int(heal_df["auto_healed"].sum()) if "auto_healed" in heal_df.columns else 0)
        col3.metric("Pending Review",
                    int((~heal_df["auto_healed"]).sum()) if "auto_healed" in heal_df.columns else 0)
        st.dataframe(heal_df, use_container_width=True)
    else:
        st.success("✅ No schema drift detected. Pipeline is healthy.")
        st.json({
            "status": "HEALTHY",
            "last_check": datetime.now(timezone.utc).isoformat(),
            "tables_monitored": ["raw_orders", "raw_events", "kpi_snapshots"],
        })


elif mode == "Pipeline Log":
    st.title("📋 Pipeline Run Log")
    if not pipeline_df.empty:
        st.dataframe(pipeline_df, use_container_width=True)
    else:
        st.info("Run the pipeline first (`python main.py --mode run`) to see logs here.")