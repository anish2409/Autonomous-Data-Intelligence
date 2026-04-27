# 🤖 ADI System — Autonomous Data Intelligence Dashboard

A production-ready Streamlit dashboard powered by a **LangGraph 3-agent pipeline** (Gemini 2.5 Flash) for real-time anomaly detection, causal analysis, and autonomous decision-making on any business dataset.

---

## 📁 Project Structure

```
project/
│
├── app1.py                  # Main Streamlit dashboard (all views + logic)
├── langgraph_agent.py       # LangGraph 3-agent pipeline (Gemini 2.5 Flash)
├── .env                     # API keys (GOOGLE_API_KEY)
│
├── local_broker.py          # Optional: LocalKafka broker for live streaming
├── db_manager.py            # Optional: AIDatabase for decision logging
├── autonomous_action.py     # Optional: ActionEngine for auto-execution
├── config/
│   └── settings.py          # Optional: DB connection config
│
└── autonomous_system.db     # SQLite audit log (auto-created at runtime)
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly sqlalchemy \
            langgraph langchain-google-genai langchain-core python-dotenv
```

### 2. Configure API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the dashboard

```bash
streamlit run app1.py
```

---

## 🧠 AI Agent Architecture — `langgraph_agent.py`

The agent pipeline is built with **LangGraph** and runs on **Gemini 2.5 Flash** (`temperature=0.2`).

### AgentState (shared memory between all agents)

```python
class AgentState(TypedDict):
    raw_data:        dict   # Input: KPI summary + anomaly list as natural language prompt
    anomaly_report:  str    # Output of Agent 1
    business_report: str    # Output of Agent 2
    final_decision:  str    # Output of Agent 3
```

### Agent Pipeline Flow

```
Input Data
    │
    ▼
┌─────────────────────────────────┐
│  Agent 1: Anomaly Expert        │  → Fraud & anomaly detection analysis
│  (anomaly_detect node)          │    Output → anomaly_report
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Agent 2: Business Strategist   │  → Business impact & root cause analysis
│  (business_analyze node)        │    Output → business_report
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Agent 3: Executive Reviewer    │  → Approve / Flag / Reject decision
│  (final_review node)            │    Output → final_decision
└─────────────────────────────────┘
    │
    ▼
 END → result returned to app1.py
```

### Quota / Error Handling

Each agent has built-in retry logic:
- On `429 RESOURCE_EXHAUSTED` → waits 10–15 seconds → retries once
- On persistent failure → returns a graceful fallback string
- `app1.py` also catches all LangGraph errors and falls back to the **rule-based council** automatically

---

## 🗂️ How `app1.py` Connects to `langgraph_agent.py`

```python
# Import — supports both root-level and agents/ subdirectory
try:
    from agents.langgraph_agent import app as agent_app
except ImportError:
    from langgraph_agent import app as agent_app
```

The dashboard calls `agent_app.invoke(payload)` where `payload` is:

```python
{
  "raw_data": """
    KPI BASELINE:
    - total_revenue: 12200.00
    - return_rate: 0.07

    DETECTED ANOMALIES:
    - total_revenue in North: observed=98450.00, expected=12200.00, z=8.94, CRITICAL
    - return_rate in West: observed=0.43, expected=0.07, z=5.12, HIGH

    Please analyze, find root causes, recommend actions.
  """
}
```

The result keys `anomaly_report`, `business_report`, `final_decision` are mapped to the dashboard's `analyst_output`, `causal_output`, `decision_output`.

### Fallback Pipeline (no API / quota exceeded)

When Gemini is unavailable, `app1.py` runs a **rule-based 3-agent council**:

| Rule-Based Agent | What it does |
|---|---|
| `_agent_analyst()` | Z-score analysis, severity breakdown, top anomalies |
| `_agent_causal()` | ATE (Average Treatment Effect) estimation, causal narratives |
| `_agent_decision()` | Priority (P1–P4), verdict, action plan |

---

## 📊 Dashboard Views

Navigate using the **sidebar radio**. All views persist across reruns via `st.session_state`.

### 🏠 Overview
- 4 KPI cards: Total Revenue, Anomalies Detected, Active Decisions, Avg Return Rate
- Hourly revenue line chart by region (48h window)
- Anomaly severity pie chart
- Latest anomalies table
- Raw CSV data explorer (when CSV uploaded)

### 🔍 Anomalies
- Filter by severity (CRITICAL / HIGH / MEDIUM / LOW) and region
- Z-score histogram
- Full anomaly table with all detected metrics

### 🔗 Causal Analysis
- **3-tier logic:**
  1. If DB connected → loads from `causal_findings` table
  2. If not → `compute_causal_impact()` using Pearson correlation on KPI data (user picks target metric)
  3. Fallback → builds causal cards from anomaly z-scores using ATE formula: `ATE = observed − expected`
- ATE bar chart (severity-coloured)
- Per-metric causal cards showing: ATE value, % deviation, z-score, confidence, root cause hypothesis

### 🤝 Agent Decisions
- Multi-agent reasoning log from DB (`agent_decisions` table)
- Per-decision expander with Analyst / Causal / Decision tabs
- SQLite audit log viewer (`autonomous_system.db`)

### 🔧 Self-Healing
- Schema drift detection log from DB (`schema_drift_log` table)
- Auto-healed vs pending review counts
- JSON health status when no drift detected

### 📋 Pipeline Log
- Run history from DB (`pipeline_runs` table)

### ⚡ Live Stream
**Purpose:** Individual anomaly events arrive one at a time → Gemini analyzes each automatically.

**How it works:**
1. Toggle **▶ Stream** ON
2. Every 3 seconds, one event is pulled from your anomaly data (cycles through all rows)
3. If **Auto-analyze** is ON → `agent_app.invoke()` is called for each event immediately
4. Right panel shows per-event Gemini output: Anomaly Report, Business Report, Final Decision
5. Manual analyze button available for unanalyzed events
6. Full decision history kept in session for up to 20 events

**Different from AI Council:** Stream = one event at a time, real-time, individual Gemini calls. Council = all anomalies together, one macro verdict.

### 🧠 AI Council
**Purpose:** All anomalies analyzed together → 3 agents debate → single consensus verdict.

**How it works:**
1. Click **▶ Run Full Council**
2. `run_ai_council()` builds a natural-language prompt with all KPI baselines and top-5 anomalies
3. `agent_app.invoke()` runs the full 3-agent LangGraph pipeline
4. Results shown in 3 tabs: Anomaly Expert | Business Strategist | Executive Reviewer
5. Verdict banner shows Priority (P1–P4), confidence score, and Final Action Plan
6. ↺ Clear Result resets so you can re-run after new data

**Priority mapping from Gemini output:**

| Keywords in `final_decision` | Priority |
|---|---|
| REJECT / CRITICAL / IMMEDIATE | 🔴 P1 |
| FLAG / URGENT / HIGH | 🟠 P2 |
| APPROVE / MONITOR / MEDIUM | 🟡 P3 |
| (default) | 🟢 P4 |

### 📊 Advanced Analytics
- Revenue share pie by region/category
- Anomaly severity distribution pie
- Average KPI bar chart by region
- Secondary KPI comparison grouped bar
- Time-series trend line chart
- Correlation heatmap (expandable)

---

## 📂 CSV Upload — Auto-Processing Engine

Upload **any CSV** via the sidebar. The system auto-detects schema — no column renaming needed.

### What happens on upload

```
Raw CSV
  │
  ▼
Column normalization (lowercase, strip special chars)
  │
  ▼
Type detection: date / numeric / categorical
  │
  ▼
Missing value fill (median for numeric, "Unknown" for categorical)
  │
  ▼
Derived metrics:
  - price × quantity → auto_revenue
  - (revenue − cost) / revenue → auto_profit_margin
  │
  ▼
Dataset type inference: sales / logs / metrics / analytics / general
  │
  ▼
Mapped to:
  - kpi_df  (snapshot_ts, region, category, total_revenue, total_orders, return_rate, avg_order_val)
  - anomaly_df  (z-score > 2.5σ flagged as anomalies, up to 50 events)
  │
  ▼
AI insight banner generated (spike detection, skew warnings, dataset type hint)
```

### Currency / percent auto-cleaning
Columns containing `₹`, `$`, `€`, `£`, `%` are stripped and converted to numeric automatically.

### Anomaly Detection Methods (sidebar selector, shown after CSV upload)

| Method | How |
|---|---|
| **Isolation Forest (ML)** | `sklearn.ensemble.IsolationForest`, contamination=5%, 100 estimators |
| **Z-Score (Fast)** | Flag rows where `|z| > 2.5` on any numeric column |

---

## 🗄️ Database Mode (Optional)

If `config/settings.py` and `db_manager.py` exist with a valid SQLAlchemy URL, the dashboard loads from these tables:

| Table | Used in view |
|---|---|
| `v_recent_kpis` | All views (KPI data) |
| `anomaly_events` | Anomalies, AI Council, Live Stream |
| `agent_decisions` | Agent Decisions view |
| `pipeline_runs` | Pipeline Log view |
| `schema_drift_log` | Self-Healing view |
| `causal_findings` | Causal Analysis view |
| `ai_audit_log` (SQLite) | Agent Decisions — live DB logs section |

When no DB is connected, the dashboard runs entirely on **built-in demo data** (48h synthetic KPI + 3 pre-defined anomalies).

---

## 🔑 Session State Keys

All state is stored in `st.session_state` and survives reruns:

| Key | Purpose |
|---|---|
| `_csv_proc` | Processed CSV dict (df, col_meta, dataset_type, etc.) |
| `_csv_kpi_df` | KPI DataFrame derived from CSV |
| `_csv_anom_df` | Anomaly DataFrame derived from CSV |
| `_csv_insight` | AI insight banner text |
| `_csv_file_name` | Uploaded filename for display |
| `_csv_file_key` | `filename_size` hash to detect new uploads |
| `_council_result` | Last AI Council run result dict |
| `_stream_council_result` | Last Live Stream council result |
| `_active_mode_idx` | Current sidebar mode index (persists across reruns) |
| `_ls_event_log` | Live Stream incoming event list (last 30) |
| `_ls_analysis_log` | Live Stream analyzed events (last 20) |
| `_ls_event_counter` | Cycles through anomaly_df rows for streaming |
| `_ls_stream_on` | Stream toggle state |
| `_ls_auto_anal` | Auto-analyze toggle state |
| `stream_logs` | Legacy decision log list |
| `stream_alerts` | Count of flagged/rejected decisions |

---

## 🎨 UI Design

- **Fonts:** Syne (headings/UI) + DM Mono (agent output boxes)
- **Background:** Multi-layer radial gradient (`#050a18` → `#04070f`) + SVG noise texture overlay
- **Accent colour:** Violet `#7c3aed` / `#a78bfa`
- **Severity colours:** Critical `#ef4444` · High `#f97316` · Medium `#eab308` · Low `#22c55e`
- **Cards:** Glassmorphism with `backdrop-filter: blur` + `inset` highlight border
- **Animations:** `fadeSlideUp` staggered entry · `heroGlow` radial pulse on hero header · `pulse` on agent status dots

---

## 🚨 Known Constraints

- Gemini `gemini-2.5-flash` has free-tier rate limits. On `429 RESOURCE_EXHAUSTED`, the system automatically falls back to the rule-based council — no crash, no data loss.
- `local_broker.py`, `db_manager.py`, `autonomous_action.py` are all **optional**. The dashboard runs fully in demo mode without them.
- SQLite audit log (`autonomous_system.db`) is only created when `ActionEngine` or `AIDatabase` is available and processes a decision.