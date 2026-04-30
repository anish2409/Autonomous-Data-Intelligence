# Autonomous Data Intelligence System (ADI)

Production-grade system for real-time data ingestion, anomaly detection, causal inference, and multi-agent decision-making — built on an e-commerce transaction dataset.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ADI SYSTEM ARCHITECTURE                           │
├────────────┬─────────────┬──────────────┬────────────┬──────────────┤
│ INGESTION  │  ANOMALY    │   CAUSAL     │   AGENTS   │ SELF-HEALING │
│            │  DETECTION  │  INFERENCE   │            │              │
│ Synthetic  │ Z-Score     │ DoWhy ATE    │ Analyst    │ Schema Drift │
│ E-commerce │ Isolation   │ OLS Fallback │ Causal     │ Auto-repair  │
│ Generator  │ Forest      │              │ Decision   │ dbt Rewrite  │
│ Bulk Write │ Ensemble    │              │ LLM Debate │              │
└────────────┴─────────────┴──────────────┴────────────┴──────────────┘
                                    │
                            PostgreSQL Store
                                    │
                         Streamlit Dashboard
```

---

## Folder Structure

```
autonomous-data-intelligence/
├── main.py                      # Entry point (seed | run | daemon | heal | demo)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── config/
│   └── settings.py              # Centralised config via env vars
│
├── sql/
│   └── schema.sql               # All tables + indexes + views
│
├── ingestion/
│   └── pipeline.py              # Data generator + bulk writer + schema drift
│
├── anomaly/
│   └── detector.py              # Z-score + Isolation Forest ensemble
│
├── causal/
│   └── inference.py             # DoWhy / OLS causal analysis
│
├── agents/
│   └── orchestrator.py          # Analyst / Causal / Decision agent debate
│
├── self_healing/
│   └── healer.py                # Schema drift repair + dbt model rewriter
│
├── dbt_models/
│   ├── dbt_project.yml
│   └── models/
│       └── kpi_snapshots.sql    # Incremental dbt model
│
├── dashboard/
│   └── app.py                   # Streamlit 5-tab dashboard
│
└── outputs/
    └── sample_output.json       # Auto-generated demo output
```

---

## Quick Start

### Option A — Demo (no database required)

```bash
git clone <repo> && cd autonomous-data-intelligence
pip install pandas numpy scikit-learn requests
python main.py --mode demo
```

### Option B — Full local setup

**1. Prerequisites**
- Python 3.11+
- PostgreSQL 14+
- (Optional) Docker + Docker Compose

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create database**
```bash
createdb adi_system
psql adi_system < sql/schema.sql
```

**4. Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials and ANTHROPIC_API_KEY
```

**5. Seed historical data (30 days)**
```bash
python main.py --mode seed
```

**6. Run a single intelligence cycle**
```bash
python main.py --mode run
```

**7. Start continuous daemon**
```bash
python main.py --mode daemon
```

**8. Launch dashboard**
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### Option C — Docker Compose

```bash
cp .env.example .env
echo "ANTHROPIC_API_KEY=your_key" >> .env

# Start all services
docker compose up -d

# Seed data (one-time)
docker compose --profile seed up adi_seed

# View logs
docker compose logs -f adi_pipeline
```

---

## Environment Variables

| Variable                  | Default         | Description                          |
|---------------------------|-----------------|--------------------------------------|
| `PGHOST`                  | localhost       | PostgreSQL host                      |
| `PGPORT`                  | 5432            | PostgreSQL port                      |
| `PGDATABASE`              | adi_system      | Database name                        |
| `PGUSER`                  | adi_user        | Database user                        |
| `PGPASSWORD`              | adi_password    | Database password                    |
| `ANTHROPIC_API_KEY`       | (empty)         | Key for LLM agent reasoning          |
| `ZSCORE_THRESHOLD`        | 3.0             | Z-score anomaly trigger              |
| `ISOLATION_CONTAMINATION` | 0.05            | Isolation Forest contamination rate  |
| `ROLLING_WINDOW_HOURS`    | 24              | Rolling window for Z-score baseline  |
| `BATCH_SIZE`              | 1000            | Orders per ingestion cycle           |
| `INGESTION_INTERVAL_SEC`  | 60              | Daemon sleep between cycles          |
| `LOG_LEVEL`               | INFO            | Python logging level                 |

---

## Module Reference

### Ingestion (`ingestion/pipeline.py`)
- `generate_orders_batch(n)` — Synthetic e-commerce orders with injected anomalies
- `run_ingestion_cycle(engine)` — Full ingest + KPI materialisation tick
- `seed_historical(engine, days)` — Populate 30 days of baseline data
- `detect_schema_drift(engine, table, df)` — Column-level drift detection

### Anomaly Detection (`anomaly/detector.py`)
- `ZScoreDetector.detect(df)` — Rolling Z-score per metric per (region, category)
- `IsolationForestDetector.detect(df)` — Multivariate ML detection
- `AnomalyDetectionEngine.run()` — Ensemble + dedup + persist

### Causal Inference (`causal/inference.py`)
- `CausalInferenceEngine.analyze(anomaly)` — Runs DoWhy or OLS over 5 causal pairs
- Output: `{cause_variable, effect_variable, ate, confidence, explanation}`

### Multi-Agent Reasoning (`agents/orchestrator.py`)
- `AgentOrchestrator.run(anomaly, causal_finding)` — 3-agent structured debate
- Agents: Analyst → Causal → Decision (with configurable debate rounds)
- Falls back to deterministic mock if no API key

### Self-Healing (`self_healing/healer.py`)
- `SelfHealingPipeline.run()` — Full drift scan + heal + validate + dbt rewrite
- Handles: NEW_COLUMN, DROPPED_COLUMN, NULL_VIOLATION, TYPE_CHANGE
- `health_report()` — 24h drift summary

---

## Sample Output

```
══════════════════════════════════════════════════════════════════════
  🤖  AUTONOMOUS DATA INTELLIGENCE SYSTEM  — DEMO OUTPUT
══════════════════════════════════════════════════════════════════════

📥  INGESTION PIPELINE
   Rows ingested: 1,000 | Status: SUCCESS

🔍  ANOMALY DETECTION (2 found)
   [CRITICAL] total_revenue
   Region/Category: North / Electronics
   Value:     98,450.75  (expected 12,200.00)
   Z-score:   8.94σ

🔗  CAUSAL INFERENCE
   avg_discount → total_revenue
   ATE: +124.5700  |  Confidence: 87%  |  Method: dowhy_backdoor_lr

🤝  MULTI-AGENT DECISIONS
   Priority: P1
   Action: Freeze discount promotions in North/Electronics for 4h
   [Decision] 1) Freeze discount promotions… 2) Alert inventory team…
              3) Activate checkout rate-limiting… 4) Notify fraud team…

🔧  SELF-HEALING
   Drifts detected: 0  |  Pipeline health: HEALTHY
```

---

## Extending the System

| Goal                        | File to modify                      |
|-----------------------------|-------------------------------------|
| Add a new KPI metric        | `anomaly/detector.py` → METRICS     |
| New causal pair hypothesis  | `causal/inference.py` → CAUSAL_PAIRS|
| Add a 4th agent             | `agents/orchestrator.py`            |
| Connect real Kafka stream   | `ingestion/pipeline.py` → swap generator |
| Add Slack alerts            | `agents/orchestrator.py` → `_persist()` |
| Change anomaly thresholds   | `.env` → ZSCORE_THRESHOLD           |

---

## Tech Stack

| Layer             | Technology                              |
|-------------------|-----------------------------------------|
| Language          | Python 3.11                             |
| Database          | PostgreSQL 16                           |
| ORM / Query       | SQLAlchemy 2.0                          |
| Data Processing   | pandas 2.2, NumPy 1.26                  |
| Anomaly Detection | scikit-learn (Isolation Forest, Z-score)|
| Causal Inference  | DoWhy 0.11, econml 0.15                 |
| LLM Agents        | Anthropic Claude API (claude-sonnet-4)  |
| Transformation    | dbt-postgres 1.8                        |
| Dashboard         | Streamlit 1.35 + Plotly                 |
| Containerisation  | Docker + Docker Compose                 |
