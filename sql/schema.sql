-- ============================================================
-- Autonomous Data Intelligence System - Schema
-- E-Commerce Transactions Dataset
-- ============================================================

-- Raw orders (source of truth)
CREATE TABLE IF NOT EXISTS raw_orders (
    order_id          BIGSERIAL PRIMARY KEY,
    customer_id       BIGINT NOT NULL,
    product_id        BIGINT NOT NULL,
    category          VARCHAR(64),
    region            VARCHAR(64),
    order_ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quantity          INT NOT NULL DEFAULT 1,
    unit_price        NUMERIC(12,2) NOT NULL,
    discount_pct      NUMERIC(5,2) DEFAULT 0,
    revenue           NUMERIC(14,2) GENERATED ALWAYS AS (
                          quantity * unit_price * (1 - discount_pct / 100)
                      ) STORED,
    payment_method    VARCHAR(32),
    is_returned       BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Raw customer events (clickstream / sessions)
CREATE TABLE IF NOT EXISTS raw_events (
    event_id     BIGSERIAL PRIMARY KEY,
    customer_id  BIGINT NOT NULL,
    session_id   UUID,
    event_type   VARCHAR(64),   -- page_view, add_to_cart, checkout, etc.
    product_id   BIGINT,
    event_ts     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata     JSONB
);

-- KPI snapshots (materialized hourly)
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    snapshot_id    BIGSERIAL PRIMARY KEY,
    snapshot_ts    TIMESTAMPTZ NOT NULL,
    period         VARCHAR(16),          -- 'hourly', 'daily'
    region         VARCHAR(64),
    category       VARCHAR(64),
    total_orders   INT,
    total_revenue  NUMERIC(16,2),
    avg_order_val  NUMERIC(12,2),
    return_rate    NUMERIC(6,4),
    conversion_rt  NUMERIC(6,4),
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Anomaly events detected by the system
CREATE TABLE IF NOT EXISTS anomaly_events (
    anomaly_id       BIGSERIAL PRIMARY KEY,
    detected_at      TIMESTAMPTZ DEFAULT NOW(),
    metric_name      VARCHAR(128),
    metric_value     NUMERIC(18,4),
    expected_value   NUMERIC(18,4),
    z_score          NUMERIC(8,4),
    isolation_score  NUMERIC(8,6),
    severity         VARCHAR(16),    -- LOW, MEDIUM, HIGH, CRITICAL
    region           VARCHAR(64),
    category         VARCHAR(64),
    window_start     TIMESTAMPTZ,
    window_end       TIMESTAMPTZ,
    raw_context      JSONB
);

-- Causal analysis results
CREATE TABLE IF NOT EXISTS causal_findings (
    finding_id       BIGSERIAL PRIMARY KEY,
    anomaly_id       BIGINT REFERENCES anomaly_events(anomaly_id),
    analyzed_at      TIMESTAMPTZ DEFAULT NOW(),
    cause_variable   VARCHAR(128),
    effect_variable  VARCHAR(128),
    ate              NUMERIC(12,6),   -- Average Treatment Effect
    confidence       NUMERIC(6,4),
    method           VARCHAR(64),
    explanation      TEXT,
    supporting_data  JSONB
);

-- Agent decisions / recommendations
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id      BIGSERIAL PRIMARY KEY,
    anomaly_id       BIGINT REFERENCES anomaly_events(anomaly_id),
    decided_at       TIMESTAMPTZ DEFAULT NOW(),
    analyst_output   TEXT,
    causal_output    TEXT,
    decision_output  TEXT,
    final_action     VARCHAR(128),
    priority         VARCHAR(16),    -- P0, P1, P2, P3
    status           VARCHAR(32) DEFAULT 'PENDING',
    assigned_to      VARCHAR(64),
    metadata         JSONB
);

-- Schema drift log
CREATE TABLE IF NOT EXISTS schema_drift_log (
    drift_id         BIGSERIAL PRIMARY KEY,
    detected_at      TIMESTAMPTZ DEFAULT NOW(),
    table_name       VARCHAR(128),
    column_name      VARCHAR(128),
    drift_type       VARCHAR(64),   -- NEW_COLUMN, TYPE_CHANGE, DROPPED_COLUMN, NULL_VIOLATION
    old_definition   TEXT,
    new_definition   TEXT,
    auto_healed      BOOLEAN DEFAULT FALSE,
    heal_query       TEXT,
    validated        BOOLEAN DEFAULT FALSE
);

-- Pipeline run audit log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           BIGSERIAL PRIMARY KEY,
    started_at       TIMESTAMPTZ DEFAULT NOW(),
    finished_at      TIMESTAMPTZ,
    pipeline_name    VARCHAR(128),
    status           VARCHAR(32),   -- RUNNING, SUCCESS, FAILED, HEALED
    rows_ingested    INT,
    anomalies_found  INT,
    error_message    TEXT,
    metrics          JSONB
);

-- ── Indexes ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_orders_ts      ON raw_orders(order_ts DESC);
CREATE INDEX IF NOT EXISTS idx_orders_region  ON raw_orders(region, order_ts DESC);
CREATE INDEX IF NOT EXISTS idx_kpi_ts         ON kpi_snapshots(snapshot_ts DESC, region, category);
CREATE INDEX IF NOT EXISTS idx_anomaly_ts     ON anomaly_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_sev    ON anomaly_events(severity, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_status   ON agent_decisions(status, decided_at DESC);

-- ── Seed data helper view ─────────────────────────────────────
CREATE OR REPLACE VIEW v_recent_kpis AS
SELECT
    k.*,
    LAG(total_revenue) OVER (PARTITION BY region, category ORDER BY snapshot_ts) AS prev_revenue,
    AVG(total_revenue) OVER (
        PARTITION BY region, category
        ORDER BY snapshot_ts
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) AS rolling_24h_avg_revenue
FROM kpi_snapshots k
WHERE period = 'hourly'
ORDER BY snapshot_ts DESC;
