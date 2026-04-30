"""
ingestion/pipeline.py
Production-grade data ingestion pipeline.
- Simulates a streaming e-commerce feed (configurable for real Kafka / CDC)
- Writes to PostgreSQL via SQLAlchemy Core (bulk-insert for throughput)
- Tracks schema of incoming data and emits drift events
- Maintains a run audit log
"""
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import config

logger = logging.getLogger("adi.ingestion")


# ── Schema registry (in-memory; extend to Redis/DB for prod) ──────────────────
_SCHEMA_REGISTRY: dict[str, dict] = {}


def get_engine() -> Engine:
    return create_engine(
        config.db.url,
        pool_size=config.db.pool_size,
        max_overflow=config.db.max_overflow,
        pool_pre_ping=True,
    )


# ── Synthetic data generator ──────────────────────────────────────────────────

CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Sports", "Books", "Beauty"]
REGIONS     = ["North", "South", "East", "West", "International"]
PAYMENTS    = ["credit_card", "debit_card", "paypal", "crypto", "bank_transfer"]

RNG = np.random.default_rng(42)


def _inject_anomalies(df: pd.DataFrame, anomaly_fraction: float = 0.02) -> pd.DataFrame:
    """Inject realistic anomalies: revenue spikes, return rate surges, discount abuse."""
    n = max(1, int(len(df) * anomaly_fraction))
    idxs = RNG.choice(df.index, size=n, replace=False)

    anomaly_type = RNG.choice(["spike", "zero", "extreme_discount"], size=n)
    for i, atype in zip(idxs, anomaly_type):
        if atype == "spike":
            df.at[i, "unit_price"] = df.at[i, "unit_price"] * RNG.uniform(8, 20)
            df.at[i, "quantity"]   = int(RNG.integers(50, 200))
        elif atype == "zero":
            df.at[i, "unit_price"] = 0.01
        else:
            df.at[i, "discount_pct"] = RNG.uniform(80, 99)

    return df


def generate_orders_batch(
    n: int = 1000,
    base_ts: Optional[datetime] = None,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate a realistic batch of e-commerce orders."""
    if base_ts is None:
        base_ts = datetime.now(timezone.utc)

    # Time spread: orders over the past ~60 seconds
    ts_offsets = RNG.integers(0, 60, size=n)
    order_ts   = [base_ts - timedelta(seconds=int(s)) for s in ts_offsets]

    df = pd.DataFrame({
        "customer_id":    RNG.integers(1_000, 500_000, size=n),
        "product_id":     RNG.integers(1, 10_000, size=n),
        "category":       RNG.choice(CATEGORIES, size=n),
        "region":         RNG.choice(REGIONS, size=n),
        "order_ts":       order_ts,
        "quantity":       RNG.integers(1, 10, size=n),
        "unit_price":     RNG.uniform(5.0, 800.0, size=n).round(2),
        "discount_pct":   RNG.choice(
                              [0, 5, 10, 15, 20, 25],
                              size=n,
                              p=[0.50, 0.20, 0.12, 0.08, 0.07, 0.03]
                          ).astype(float),
        "payment_method": RNG.choice(PAYMENTS, size=n),
        "is_returned":    RNG.choice([False, True], size=n, p=[0.93, 0.07]),
    })

    if inject_anomalies:
        df = _inject_anomalies(df)

    return df


def generate_events_batch(n: int = 3000) -> pd.DataFrame:
    """Generate raw clickstream events."""
    event_types = ["page_view", "add_to_cart", "remove_from_cart", "checkout", "purchase"]
    probs       = [0.55, 0.20, 0.08, 0.10, 0.07]

    return pd.DataFrame({
        "customer_id": RNG.integers(1_000, 500_000, size=n),
        "session_id":  [str(uuid.uuid4()) for _ in range(n)],
        "event_type":  RNG.choice(event_types, size=n, p=probs),
        "product_id":  RNG.integers(1, 10_000, size=n),
        "event_ts":    [datetime.now(timezone.utc) - timedelta(seconds=int(s))
                        for s in RNG.integers(0, 120, size=n)],
        "metadata":    [None] * n,   # JSON in prod
    })


# ── Schema drift detection ─────────────────────────────────────────────────────

def capture_schema(engine: Engine, table_name: str) -> dict:
    """Return column -> type mapping for a live table."""
    insp = inspect(engine)
    try:
        cols = insp.get_columns(table_name)
        return {c["name"]: str(c["type"]) for c in cols}
    except Exception:
        return {}


def detect_schema_drift(engine: Engine, table_name: str, incoming_df: pd.DataFrame) -> list[dict]:
    """Compare incoming DataFrame columns against registered schema."""
    live_schema = capture_schema(engine, table_name)
    known        = _SCHEMA_REGISTRY.get(table_name, live_schema)
    _SCHEMA_REGISTRY[table_name] = live_schema

    drifts = []
    incoming_cols = set(incoming_df.columns)
    known_cols    = set(known.keys())

    for col in incoming_cols - known_cols:
        drifts.append({
            "table_name":    table_name,
            "column_name":   col,
            "drift_type":    "NEW_COLUMN",
            "old_definition": None,
            "new_definition": str(incoming_df[col].dtype),
        })
    for col in known_cols - incoming_cols:
        drifts.append({
            "table_name":    table_name,
            "column_name":   col,
            "drift_type":    "DROPPED_COLUMN",
            "old_definition": known[col],
            "new_definition": None,
        })

    return drifts


def log_drift_events(engine: Engine, drifts: list[dict]) -> None:
    if not drifts:
        return
    with engine.begin() as conn:
        for d in drifts:
            conn.execute(text("""
                INSERT INTO schema_drift_log
                    (table_name, column_name, drift_type, old_definition, new_definition)
                VALUES
                    (:table_name, :column_name, :drift_type, :old_definition, :new_definition)
            """), d)
    logger.warning("Schema drift logged: %d events", len(drifts))


# ── KPI materializer ──────────────────────────────────────────────────────────

def materialize_kpi_snapshot(engine: Engine) -> int:
    """Aggregate last hour of orders into kpi_snapshots."""
    sql = """
    INSERT INTO kpi_snapshots
        (snapshot_ts, period, region, category,
         total_orders, total_revenue, avg_order_val, return_rate, conversion_rt)
    SELECT
        date_trunc('hour', NOW())                           AS snapshot_ts,
        'hourly'                                            AS period,
        region,
        category,
        COUNT(*)                                            AS total_orders,
        COALESCE(SUM(revenue), 0)                           AS total_revenue,
        COALESCE(AVG(revenue), 0)                           AS avg_order_val,
        COALESCE(AVG(is_returned::int), 0)                  AS return_rate,
        0.05 + RANDOM() * 0.10                              AS conversion_rt   -- placeholder
    FROM raw_orders
    WHERE order_ts >= date_trunc('hour', NOW()) - INTERVAL '1 hour'
      AND order_ts <  date_trunc('hour', NOW())
    GROUP BY region, category
    ON CONFLICT DO NOTHING
    RETURNING 1
    """
    with engine.begin() as conn:
        result = conn.execute(text(sql))
        rows = result.rowcount
    logger.info("KPI snapshot: %d rows materialized", rows)
    return rows


# ── Bulk write helpers ─────────────────────────────────────────────────────────

def bulk_write(engine: Engine, df: pd.DataFrame, table: str, run_id: int) -> int:
    """Write DataFrame to PostgreSQL using multi-row INSERT (fast path)."""
    if df.empty:
        return 0
    try:
        df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info("[run=%d] Wrote %d rows → %s", run_id, len(df), table)
        return len(df)
    except SQLAlchemyError as exc:
        logger.error("[run=%d] Bulk write failed on %s: %s", run_id, table, exc)
        raise


def start_run(engine: Engine, pipeline_name: str) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO pipeline_runs (pipeline_name, status)
            VALUES (:name, 'RUNNING') RETURNING run_id
        """), {"name": pipeline_name}).fetchone()
    return row[0]


def finish_run(engine: Engine, run_id: int, status: str,
               rows: int = 0, anomalies: int = 0, error: str = None) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pipeline_runs
            SET finished_at = NOW(), status = :status,
                rows_ingested = :rows, anomalies_found = :anomalies,
                error_message = :error
            WHERE run_id = :run_id
        """), {"status": status, "rows": rows, "anomalies": anomalies,
               "error": error, "run_id": run_id})


# ── Main ingestion loop ────────────────────────────────────────────────────────

def run_ingestion_cycle(engine: Engine, batch_n: int = 1000) -> dict:
    """One ingestion tick: generate → validate → write → snapshot."""
    run_id      = start_run(engine, "ecommerce_orders")
    total_rows  = 0
    anomalies   = 0

    try:
        # 1. Generate batches
        orders_df = generate_orders_batch(n=batch_n)
        events_df = generate_events_batch(n=batch_n * 3)

        # 2. Schema drift check
        drifts = detect_schema_drift(engine, "raw_orders", orders_df)
        log_drift_events(engine, drifts)

        # 3. Persist
        total_rows += bulk_write(engine, orders_df, "raw_orders", run_id)
        bulk_write(engine, events_df, "raw_events", run_id)

        # 4. Materialize KPIs
        materialize_kpi_snapshot(engine)

        finish_run(engine, run_id, "SUCCESS", total_rows, anomalies)
        return {"run_id": run_id, "rows": total_rows, "status": "SUCCESS"}

    except Exception as exc:
        finish_run(engine, run_id, "FAILED", total_rows, 0, str(exc))
        logger.exception("Ingestion cycle failed: %s", exc)
        return {"run_id": run_id, "rows": total_rows, "status": "FAILED", "error": str(exc)}


def seed_historical(engine: Engine, days: int = 30, rows_per_hour: int = 500) -> None:
    """Seed historical data so anomaly detection has enough baseline."""
    logger.info("Seeding %d days of historical orders…", days)
    now = datetime.now(timezone.utc)
    total = 0
    for h in range(days * 24, 0, -1):
        ts  = now - timedelta(hours=h)
        df  = generate_orders_batch(n=rows_per_hour, base_ts=ts, inject_anomalies=(h % 12 == 0))
        run = start_run(engine, "seed_historical")
        n   = bulk_write(engine, df, "raw_orders", run)
        finish_run(engine, run, "SUCCESS", n)
        total += n
        if h % 24 == 0:
            logger.info("Seeded %d rows so far…", total)
    materialize_kpi_snapshot(engine)
    logger.info("Seeding complete: %d total rows", total)
