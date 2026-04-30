"""
monitoring/alerting.py
Production alerting layer for the ADI System.

Supports:
  - Slack webhooks
  - PagerDuty Events API v2
  - Email via SMTP
  - Structured alert log (always writes to DB + local file)

Priority routing:
  P0/CRITICAL → PagerDuty + Slack + Email
  P1/HIGH     → Slack + Email
  P2/MEDIUM   → Slack
  P3/LOW      → log only
"""
import json
import logging
import os
import smtplib
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests

logger = logging.getLogger("adi.alerting")


# ── Config (all from env) ─────────────────────────────────────────────────────

SLACK_WEBHOOK_URL   = os.getenv("SLACK_WEBHOOK_URL", "")
PAGERDUTY_KEY       = os.getenv("PAGERDUTY_ROUTING_KEY", "")
SMTP_HOST           = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT           = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER           = os.getenv("SMTP_USER", "")
SMTP_PASSWORD       = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO      = os.getenv("ALERT_EMAIL_TO", "oncall@company.com")
ALERT_LOG_PATH      = os.getenv("ALERT_LOG_PATH", "outputs/alerts.jsonl")


# ── Alert data class ──────────────────────────────────────────────────────────

@dataclass
class Alert:
    title:       str
    body:        str
    priority:    str            # P0 | P1 | P2 | P3
    severity:    str            # CRITICAL | HIGH | MEDIUM | LOW
    anomaly_id:  Optional[int]
    metric_name: str
    region:      str
    category:    str
    z_score:     float
    action:      str
    source:      str = "ADI System"
    fired_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    channels:    list[str] = field(default_factory=list)
    errors:      list[str] = field(default_factory=list)


# ── Channel implementations ───────────────────────────────────────────────────

def _send_slack(alert: Alert) -> bool:
    if not SLACK_WEBHOOK_URL:
        logger.debug("Slack webhook not configured; skipping")
        return False

    color_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316",
                 "MEDIUM": "#eab308", "LOW": "#22c55e"}
    color     = color_map.get(alert.severity, "#6b7280")
    emoji     = {"P0": "🚨", "P1": "🔴", "P2": "🟡", "P3": "🟢"}.get(alert.priority, "⚪")

    payload = {
        "text": f"{emoji} *[{alert.priority}] {alert.title}*",
        "attachments": [{
            "color": color,
            "fields": [
                {"title": "Metric",    "value": alert.metric_name, "short": True},
                {"title": "Region",    "value": alert.region,      "short": True},
                {"title": "Category",  "value": alert.category,    "short": True},
                {"title": "Z-score",   "value": f"{alert.z_score:.2f}σ", "short": True},
                {"title": "Anomaly ID","value": str(alert.anomaly_id), "short": True},
                {"title": "Fired At",  "value": alert.fired_at,    "short": True},
                {"title": "Action",    "value": alert.action, "short": False},
            ],
            "footer": "ADI System | Autonomous Data Intelligence",
        }],
    }
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Slack alert sent (priority=%s)", alert.priority)
        return True
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)
        alert.errors.append(f"slack:{exc}")
        return False


def _send_pagerduty(alert: Alert) -> bool:
    if not PAGERDUTY_KEY:
        logger.debug("PagerDuty key not configured; skipping")
        return False

    severity_map = {"CRITICAL": "critical", "HIGH": "error",
                    "MEDIUM": "warning", "LOW": "info"}
    payload = {
        "routing_key":  PAGERDUTY_KEY,
        "event_action": "trigger",
        "dedup_key":    f"adi-anomaly-{alert.anomaly_id}-{alert.metric_name}",
        "payload": {
            "summary":   alert.title,
            "severity":  severity_map.get(alert.severity, "warning"),
            "source":    alert.source,
            "timestamp": alert.fired_at,
            "custom_details": {
                "body":       alert.body,
                "action":     alert.action,
                "region":     alert.region,
                "category":   alert.category,
                "z_score":    alert.z_score,
                "anomaly_id": alert.anomaly_id,
            },
        },
    }
    try:
        r = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload, timeout=15
        )
        r.raise_for_status()
        logger.info("PagerDuty alert triggered (key=%s…)", PAGERDUTY_KEY[:8])
        return True
    except Exception as exc:
        logger.warning("PagerDuty alert failed: %s", exc)
        alert.errors.append(f"pagerduty:{exc}")
        return False


def _send_email(alert: Alert) -> bool:
    if not (SMTP_USER and SMTP_PASSWORD):
        logger.debug("SMTP credentials not configured; skipping email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.priority}] ADI Alert: {alert.title}"
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_EMAIL_TO

        html = f"""
        <html><body style="font-family:sans-serif;max-width:600px">
        <h2 style="color:{'#ef4444' if alert.severity=='CRITICAL' else '#f97316'}">
            [{alert.priority}] {alert.title}
        </h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">
            <tr><th>Metric</th><td>{alert.metric_name}</td></tr>
            <tr><th>Region / Category</th><td>{alert.region} / {alert.category}</td></tr>
            <tr><th>Z-score</th><td>{alert.z_score:.2f}σ</td></tr>
            <tr><th>Anomaly ID</th><td>{alert.anomaly_id}</td></tr>
            <tr><th>Fired At</th><td>{alert.fired_at}</td></tr>
        </table>
        <h3>Description</h3><p>{alert.body}</p>
        <h3>Recommended Action</h3>
        <p style="background:#f3f4f6;padding:12px;border-radius:6px">{alert.action}</p>
        <hr/><small>ADI System — Autonomous Data Intelligence</small>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(SMTP_USER, ALERT_EMAIL_TO, msg.as_string())

        logger.info("Email alert sent to %s", ALERT_EMAIL_TO)
        return True
    except Exception as exc:
        logger.warning("Email alert failed: %s", exc)
        alert.errors.append(f"email:{exc}")
        return False


def _write_alert_log(alert: Alert) -> None:
    """Always write to a local JSONL file regardless of channel success."""
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(alert), default=str) + "\n")


def _persist_alert_db(alert: Alert, engine) -> None:
    """Optional: store alert in agent_decisions status update."""
    if engine is None or alert.anomaly_id is None:
        return
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE agent_decisions
                SET status = 'ALERTED', metadata = metadata || :patch
                WHERE anomaly_id = :aid
            """), {
                "patch": json.dumps({"alert_channels": alert.channels,
                                     "alert_fired_at": alert.fired_at}),
                "aid":   alert.anomaly_id,
            })
    except Exception as exc:
        logger.debug("DB alert persistence failed (non-critical): %s", exc)


# ── Routing logic ─────────────────────────────────────────────────────────────

ROUTING: dict[str, list[str]] = {
    "P0": ["pagerduty", "slack", "email"],
    "P1": ["slack", "email"],
    "P2": ["slack"],
    "P3": [],
}


def fire_alert(
    anomaly:  dict,
    decision: dict,
    engine=None,
) -> Alert:
    """
    Main entry: build and dispatch an alert from anomaly + agent decision dicts.

    anomaly:  from anomaly_events table
    decision: from agent_decisions table / AgentDebateResult
    """
    priority = decision.get("priority", "P2")
    severity = anomaly.get("severity", "MEDIUM")

    title = (
        f"{severity} anomaly in {anomaly.get('region')}/{anomaly.get('category')}: "
        f"{anomaly.get('metric_name')} deviated {anomaly.get('z_score', 0):.1f}σ"
    )
    body  = (
        f"Detected at {anomaly.get('detected_at')}.\n"
        f"Value: {anomaly.get('metric_value')} (expected {anomaly.get('expected_value')}).\n"
        f"Agent assessment: {decision.get('analyst_output', '')[:300]}"
    )

    alert = Alert(
        title=title,
        body=body,
        priority=priority,
        severity=severity,
        anomaly_id=anomaly.get("anomaly_id"),
        metric_name=anomaly.get("metric_name", ""),
        region=anomaly.get("region", ""),
        category=anomaly.get("category", ""),
        z_score=float(anomaly.get("z_score", 0)),
        action=decision.get("final_action", "Review manually"),
    )

    channels = ROUTING.get(priority, [])
    logger.info("Firing alert [%s] via channels: %s", priority, channels)

    for channel in channels:
        ok = False
        if channel == "slack":      ok = _send_slack(alert)
        if channel == "pagerduty":  ok = _send_pagerduty(alert)
        if channel == "email":      ok = _send_email(alert)
        if ok:
            alert.channels.append(channel)

    _write_alert_log(alert)
    _persist_alert_db(alert, engine)

    return alert


# ── Health / metrics probe ────────────────────────────────────────────────────

def alert_summary(log_path: str = ALERT_LOG_PATH) -> dict:
    """Read alert log and return summary stats."""
    if not os.path.exists(log_path):
        return {"total": 0, "by_priority": {}, "by_severity": {}}

    alerts     = []
    by_priority = {}
    by_severity = {}

    with open(log_path) as f:
        for line in f:
            try:
                a = json.loads(line)
                alerts.append(a)
                p = a.get("priority", "?")
                s = a.get("severity", "?")
                by_priority[p] = by_priority.get(p, 0) + 1
                by_severity[s] = by_severity.get(s, 0) + 1
            except json.JSONDecodeError:
                pass

    return {
        "total":        len(alerts),
        "by_priority":  by_priority,
        "by_severity":  by_severity,
        "latest":       alerts[-1] if alerts else None,
    }
