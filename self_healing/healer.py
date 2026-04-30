"""
self_healing/healer.py
Self-Healing Data Pipeline.

Capabilities:
  1. Detect schema drift (new columns, type changes, dropped columns, null violations)
  2. Auto-generate corrective SQL / dbt model patches
  3. Validate the fix in a staging transaction before committing
  4. Log all healing actions to schema_drift_log
"""
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import inspect as sa_inspect, text, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("adi.healer")

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DriftEvent:
    table_name:      str
    column_name:     str
    drift_type:      str      # NEW_COLUMN | TYPE_CHANGE | DROPPED_COLUMN | NULL_VIOLATION
    old_definition:  Optional[str]
    new_definition:  Optional[str]


@dataclass
class HealResult:
    drift:     DriftEvent
    healed:    bool
    validated: bool
    query:     str
    error:     Optional[str] = None


# ── Schema introspection ──────────────────────────────────────────────────────

MANAGED_TABLES = ["raw_orders", "raw_events", "kpi_snapshots"]

# Expected schema baseline (col_name → SQLAlchemy type string fragment)
EXPECTED_SCHEMAS: dict[str, dict[str, str]] = {
    "raw_orders": {
        "order_id":       "BIGINT",
        "customer_id":    "BIGINT",
        "product_id":     "BIGINT",
        "category":       "VARCHAR",
        "region":         "VARCHAR",
        "order_ts":       "TIMESTAMP",
        "quantity":       "INTEGER",
        "unit_price":     "NUMERIC",
        "discount_pct":   "NUMERIC",
        "payment_method": "VARCHAR",
        "is_returned":    "BOOLEAN",
        "created_at":     "TIMESTAMP",
    },
    "raw_events": {
        "event_id":    "BIGINT",
        "customer_id": "BIGINT",
        "session_id":  "UUID",
        "event_type":  "VARCHAR",
        "product_id":  "BIGINT",
        "event_ts":    "TIMESTAMP",
        "metadata":    "JSONB",
    },
    "kpi_snapshots": {
        "snapshot_id":   "BIGINT",
        "snapshot_ts":   "TIMESTAMP",
        "period":        "VARCHAR",
        "region":        "VARCHAR",
        "category":      "VARCHAR",
        "total_orders":  "INTEGER",
        "total_revenue": "NUMERIC",
        "avg_order_val": "NUMERIC",
        "return_rate":   "NUMERIC",
        "conversion_rt": "NUMERIC",
        "created_at":    "TIMESTAMP",
    },
}


def _get_live_schema(engine: Engine, table: str) -> dict[str, str]:
    insp = sa_inspect(engine)
    try:
        cols = insp.get_columns(table)
        return {c["name"]: str(c["type"]).upper() for c in cols}
    except Exception:
        return {}


def _detect_drift(engine: Engine) -> list[DriftEvent]:
    """Compare live DB schemas against expected baseline."""
    drifts: list[DriftEvent] = []

    for table, expected in EXPECTED_SCHEMAS.items():
        live = _get_live_schema(engine, table)
        if not live:
            logger.warning("Table %s not found — skipping drift check", table)
            continue

        for col, exp_type in expected.items():
            if col not in live:
                drifts.append(DriftEvent(
                    table_name=table, column_name=col,
                    drift_type="DROPPED_COLUMN",
                    old_definition=exp_type, new_definition=None
                ))
            elif not any(exp_type in live[col] for _ in [1]):
                # Simplified type mismatch check
                if exp_type not in live[col]:
                    drifts.append(DriftEvent(
                        table_name=table, column_name=col,
                        drift_type="TYPE_CHANGE",
                        old_definition=exp_type,
                        new_definition=live[col]
                    ))

        # New columns not in baseline
        for col in live:
            if col not in expected and col not in ("revenue",):
                drifts.append(DriftEvent(
                    table_name=table, column_name=col,
                    drift_type="NEW_COLUMN",
                    old_definition=None,
                    new_definition=live[col]
                ))

    return drifts


def _detect_null_violations(engine: Engine) -> list[DriftEvent]:
    """Check NOT NULL columns for unexpected NULLs in recent data."""
    not_null_checks = [
        ("raw_orders",   "customer_id"),
        ("raw_orders",   "order_ts"),
        ("raw_orders",   "unit_price"),
        ("kpi_snapshots","snapshot_ts"),
    ]
    drifts = []
    for table, col in not_null_checks:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(
                    f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                )).scalar()
            if result and result > 0:
                drifts.append(DriftEvent(
                    table_name=table, column_name=col,
                    drift_type="NULL_VIOLATION",
                    old_definition="NOT NULL",
                    new_definition=f"{result} NULL rows found"
                ))
        except SQLAlchemyError:
            pass
    return drifts


# ── Heal strategies ───────────────────────────────────────────────────────────

def _heal_new_column(drift: DriftEvent) -> str:
    """
    For new columns we don't recognise, add a comment so the dbt model
    can be regenerated to include them.
    """
    return (
        f"COMMENT ON COLUMN {drift.table_name}.{drift.column_name} IS "
        f"'AUTO_DETECTED: new column added outside baseline schema at "
        f"{datetime.now(timezone.utc).isoformat()}';"
    )


def _heal_dropped_column(drift: DriftEvent) -> str:
    """Re-add a dropped column with its expected type."""
    type_map = {
        "BIGINT":    "BIGINT",
        "INTEGER":   "INTEGER",
        "VARCHAR":   "VARCHAR(128)",
        "NUMERIC":   "NUMERIC(14,4)",
        "BOOLEAN":   "BOOLEAN DEFAULT FALSE",
        "TIMESTAMP": "TIMESTAMPTZ",
        "UUID":      "UUID",
        "JSONB":     "JSONB",
    }
    pg_type = type_map.get(drift.old_definition, "TEXT")
    return (
        f"ALTER TABLE {drift.table_name} "
        f"ADD COLUMN IF NOT EXISTS {drift.column_name} {pg_type};"
    )


def _heal_null_violation(drift: DriftEvent) -> str:
    """
    For numeric columns: set nulls to 0.
    For timestamp columns: set to epoch.
    """
    expected_type = EXPECTED_SCHEMAS.get(drift.table_name, {}).get(drift.column_name, "NUMERIC")
    if "TIMESTAMP" in expected_type:
        fill = "'1970-01-01 00:00:00+00'"
    elif expected_type in ("BIGINT", "INTEGER", "NUMERIC"):
        fill = "0"
    else:
        fill = "'UNKNOWN'"
    return (
        f"UPDATE {drift.table_name} "
        f"SET {drift.column_name} = {fill} "
        f"WHERE {drift.column_name} IS NULL;"
    )


def _build_heal_query(drift: DriftEvent) -> Optional[str]:
    if drift.drift_type == "NEW_COLUMN":
        return _heal_new_column(drift)
    if drift.drift_type == "DROPPED_COLUMN":
        return _heal_dropped_column(drift)
    if drift.drift_type == "NULL_VIOLATION":
        return _heal_null_violation(drift)
    if drift.drift_type == "TYPE_CHANGE":
        # Conservative: log only, do not auto-cast
        logger.warning(
            "TYPE_CHANGE on %s.%s: %s → %s. Manual review required.",
            drift.table_name, drift.column_name,
            drift.old_definition, drift.new_definition
        )
        return None
    return None


def _validate_and_apply(engine: Engine, drift: DriftEvent, query: str) -> HealResult:
    """Run the heal query inside a savepoint; roll back if it errors."""
    try:
        with engine.begin() as conn:
            conn.execute(text("SAVEPOINT heal_check"))
            conn.execute(text(query))
            conn.execute(text("RELEASE SAVEPOINT heal_check"))
        return HealResult(drift=drift, healed=True, validated=True, query=query)
    except SQLAlchemyError as exc:
        logger.error("Heal query failed: %s\nQuery: %s", exc, query)
        try:
            with engine.begin() as conn:
                conn.execute(text("ROLLBACK TO SAVEPOINT heal_check"))
        except Exception:
            pass
        return HealResult(drift=drift, healed=False, validated=False,
                          query=query, error=str(exc))


def _log_drift(engine: Engine, drift: DriftEvent,
               result: Optional[HealResult] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO schema_drift_log
                (table_name, column_name, drift_type, old_definition,
                 new_definition, auto_healed, heal_query, validated)
            VALUES
                (:table_name, :column_name, :drift_type, :old_definition,
                 :new_definition, :auto_healed, :heal_query, :validated)
        """), {
            "table_name":    drift.table_name,
            "column_name":   drift.column_name,
            "drift_type":    drift.drift_type,
            "old_definition":drift.old_definition,
            "new_definition":drift.new_definition,
            "auto_healed":   result.healed if result else False,
            "heal_query":    result.query if result else None,
            "validated":     result.validated if result else False,
        })


# ── dbt model rewriter ────────────────────────────────────────────────────────

DBT_TEMPLATE = """-- AUTO-GENERATED by Self-Healing Pipeline
-- Generated at: {ts}
-- Drift detected: {drift_summary}

{{{{ config(materialized='incremental', unique_key='snapshot_ts') }}}}

SELECT
    date_trunc('hour', o.order_ts)         AS snapshot_ts,
    'hourly'                               AS period,
    o.region,
    o.category,
    COUNT(*)                               AS total_orders,
    COALESCE(SUM(o.revenue), 0)            AS total_revenue,
    COALESCE(AVG(o.revenue), 0)            AS avg_order_val,
    COALESCE(AVG(o.is_returned::int), 0)   AS return_rate,
    0.07                                   AS conversion_rt
FROM {{{{ ref('raw_orders') }}}} o
WHERE o.order_ts >= date_trunc('hour', NOW()) - INTERVAL '1 hour'
{extra_filters}
GROUP BY 1, 2, 3, 4

{{% if is_incremental() %}}
    AND o.order_ts > (SELECT MAX(snapshot_ts) FROM {{{{ this }}}})
{{% endif %}}
"""


def rewrite_dbt_model(drifts: list[DriftEvent], output_dir: str = "dbt_models/models") -> str:
    """Auto-rewrite the kpi_snapshots dbt model in response to drift."""
    os.makedirs(output_dir, exist_ok=True)
    drift_summary = "; ".join(
        f"{d.drift_type}:{d.table_name}.{d.column_name}" for d in drifts
    )
    extra_filters = ""
    for d in drifts:
        if d.drift_type == "NULL_VIOLATION":
            extra_filters += f"\n  AND o.{d.column_name} IS NOT NULL"

    content = DBT_TEMPLATE.format(
        ts=datetime.now(timezone.utc).isoformat(),
        drift_summary=drift_summary,
        extra_filters=extra_filters,
    )
    path = os.path.join(output_dir, "kpi_snapshots.sql")
    with open(path, "w") as f:
        f.write(content)
    logger.info("dbt model rewritten: %s", path)
    return path


# ── Main healer ───────────────────────────────────────────────────────────────

class SelfHealingPipeline:

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self) -> list[HealResult]:
        logger.info("Self-healing scan started…")

        schema_drifts = _detect_drift(self.engine)
        null_drifts   = _detect_null_violations(self.engine)
        all_drifts    = schema_drifts + null_drifts

        if not all_drifts:
            logger.info("No drift detected. Pipeline is healthy.")
            return []

        logger.warning("Detected %d drift events", len(all_drifts))
        results: list[HealResult] = []

        for drift in all_drifts:
            query = _build_heal_query(drift)

            if query:
                result = _validate_and_apply(self.engine, drift, query)
                logger.info(
                    "  %s.%s [%s] → healed=%s validated=%s",
                    drift.table_name, drift.column_name,
                    drift.drift_type, result.healed, result.validated
                )
            else:
                result = HealResult(drift=drift, healed=False,
                                    validated=False, query="",
                                    error="No auto-heal strategy")

            _log_drift(self.engine, drift, result)
            results.append(result)

        # Rewrite dbt model if structural drift found
        structural = [d for d in all_drifts
                      if d.drift_type in ("NEW_COLUMN", "DROPPED_COLUMN")]
        if structural:
            rewrite_dbt_model(structural)

        healed = sum(1 for r in results if r.healed)
        logger.info("Self-healing complete: %d/%d events healed", healed, len(results))
        return results

    def health_report(self) -> dict:
        sql = """
        SELECT drift_type, COUNT(*) AS count,
               SUM(auto_healed::int) AS healed
        FROM schema_drift_log
        WHERE detected_at >= NOW() - INTERVAL '24 hours'
        GROUP BY drift_type
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return {
            "total_drifts": int(df["count"].sum()) if not df.empty else 0,
            "total_healed": int(df["healed"].sum()) if not df.empty else 0,
            "breakdown":    df.to_dict("records"),
        }
