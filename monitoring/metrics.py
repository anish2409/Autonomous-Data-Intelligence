"""
monitoring/metrics.py
Prometheus-compatible metrics exporter + lightweight /health HTTP endpoint.

Exposes:
  adi_anomalies_total{severity, region, category}
  adi_pipeline_rows_ingested_total
  adi_pipeline_run_duration_seconds
  adi_schema_drifts_total{drift_type}
  adi_agent_decisions_total{priority}
  adi_system_health (1 = healthy)

Run standalone:
  python monitoring/metrics.py
  curl http://localhost:8000/metrics
  curl http://localhost:8000/health
"""
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("adi.metrics")

# Optional prometheus_client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed; using text fallback")


# ── Metric definitions (only if prometheus_client available) ──────────────────

if PROM_AVAILABLE:
    REGISTRY = CollectorRegistry(auto_describe=True)

    ANOMALIES_TOTAL = Counter(
        "adi_anomalies_total",
        "Total anomalies detected",
        ["severity", "region", "category"],
        registry=REGISTRY,
    )
    ROWS_INGESTED = Counter(
        "adi_pipeline_rows_ingested_total",
        "Total rows ingested by the pipeline",
        registry=REGISTRY,
    )
    RUN_DURATION = Histogram(
        "adi_pipeline_run_duration_seconds",
        "Duration of a full pipeline cycle",
        buckets=[1, 5, 10, 30, 60, 120, 300],
        registry=REGISTRY,
    )
    SCHEMA_DRIFTS = Counter(
        "adi_schema_drifts_total",
        "Schema drift events detected",
        ["drift_type"],
        registry=REGISTRY,
    )
    AGENT_DECISIONS = Counter(
        "adi_agent_decisions_total",
        "Agent decisions produced",
        ["priority"],
        registry=REGISTRY,
    )
    SYSTEM_HEALTH = Gauge(
        "adi_system_health",
        "1 = system healthy, 0 = degraded",
        registry=REGISTRY,
    )
    SYSTEM_HEALTH.set(1)


# ── Recording helpers (called from main pipeline) ─────────────────────────────

def record_anomaly(severity: str, region: str, category: str) -> None:
    if PROM_AVAILABLE:
        ANOMALIES_TOTAL.labels(severity=severity, region=region, category=category).inc()


def record_rows_ingested(n: int) -> None:
    if PROM_AVAILABLE:
        ROWS_INGESTED.inc(n)


def record_run_duration(seconds: float) -> None:
    if PROM_AVAILABLE:
        RUN_DURATION.observe(seconds)


def record_drift(drift_type: str) -> None:
    if PROM_AVAILABLE:
        SCHEMA_DRIFTS.labels(drift_type=drift_type).inc()


def record_decision(priority: str) -> None:
    if PROM_AVAILABLE:
        AGENT_DECISIONS.labels(priority=priority).inc()


def set_health(healthy: bool) -> None:
    if PROM_AVAILABLE:
        SYSTEM_HEALTH.set(1 if healthy else 0)


# ── In-memory fallback counters (when prometheus_client absent) ───────────────

_FALLBACK: dict[str, float] = {
    "anomalies_total":    0,
    "rows_ingested_total": 0,
    "schema_drifts_total": 0,
    "agent_decisions_total": 0,
    "system_health": 1,
}

def _fallback_text() -> str:
    lines = ["# ADI System Metrics (fallback, no prometheus_client)"]
    for k, v in _FALLBACK.items():
        lines.append(f"adi_{k} {v}")
    return "\n".join(lines) + "\n"


# ── Lightweight DB probe ──────────────────────────────────────────────────────

def _db_health(engine) -> dict:
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _pipeline_health(engine) -> dict:
    try:
        from sqlalchemy import text
        import pandas as pd
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT status, COUNT(*) AS n
                FROM pipeline_runs
                WHERE started_at >= NOW() - INTERVAL '1 hour'
                GROUP BY status
            """), conn)
        return df.set_index("status")["n"].to_dict()
    except Exception:
        return {}


def build_health_report(engine=None) -> dict:
    db = _db_health(engine) if engine else {"status": "not_configured"}
    pl = _pipeline_health(engine) if engine else {}

    healthy = db["status"] == "ok"
    set_health(healthy)

    return {
        "status":           "healthy" if healthy else "degraded",
        "timestamp":        __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "database":         db,
        "pipeline_last_1h": pl,
        "metrics_backend":  "prometheus_client" if PROM_AVAILABLE else "fallback",
    }


# ── HTTP server ───────────────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):

    engine: Optional[object] = None  # injected at server startup

    def log_message(self, fmt, *args):
        pass   # suppress default HTTP server logs

    def do_GET(self):
        if self.path == "/metrics":
            self._metrics()
        elif self.path in ("/health", "/healthz", "/"):
            self._health()
        else:
            self.send_response(404)
            self.end_headers()

    def _metrics(self):
        if PROM_AVAILABLE:
            body = generate_latest(REGISTRY)
            ctype = CONTENT_TYPE_LATEST
        else:
            body = _fallback_text().encode()
            ctype = "text/plain; version=0.0.4"

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def _health(self):
        report = build_health_report(self.engine)
        body   = json.dumps(report, indent=2).encode()
        code   = 200 if report["status"] == "healthy" else 503

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def start_metrics_server(port: int = 8000, engine=None) -> HTTPServer:
    MetricsHandler.engine = engine
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    logger.info("Metrics server started on http://0.0.0.0:%d", port)
    return server


if __name__ == "__main__":
    import threading
    logging.basicConfig(level="INFO")
    server = start_metrics_server(port=int(os.getenv("METRICS_PORT", "8000")))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print("Metrics: http://localhost:8000/metrics")
    print("Health:  http://localhost:8000/health")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
