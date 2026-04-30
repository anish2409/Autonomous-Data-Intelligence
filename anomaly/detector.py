"""
anomaly/detector.py
Statistical + ML anomaly detection engine.
- Z-score on rolling KPI windows
- Isolation Forest for multivariate anomalies
- Severity classification
- Persistence to anomaly_events table
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import config

logger = logging.getLogger("adi.anomaly")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    metric_name:      str
    metric_value:     float
    expected_value:   float
    z_score:          float
    isolation_score:  float
    severity:         str
    region:           str
    category:         str
    window_start:     Optional[datetime]
    window_end:       Optional[datetime]
    raw_context:      dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_severity(z: float) -> str:
    az = abs(z)
    for level, (lo, hi) in config.anomaly.severity_thresholds.items():
        if lo <= az < hi:
            return level
    return "CRITICAL"


def _load_kpi_window(engine: Engine, hours: int = 48) -> pd.DataFrame:
    sql = f"""
    SELECT
        snapshot_ts, region, category,
        total_orders, total_revenue, avg_order_val,
        return_rate, conversion_rt
    FROM kpi_snapshots
    WHERE period = 'hourly'
      AND snapshot_ts >= NOW() - INTERVAL '{hours} hours'
    ORDER BY snapshot_ts ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, parse_dates=["snapshot_ts"])
    return df


# ── Statistical: Z-score ──────────────────────────────────────────────────────

class ZScoreDetector:
    """Per-group rolling Z-score for each KPI metric."""

    METRICS = ["total_revenue", "avg_order_val", "return_rate", "total_orders"]

    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.anomaly.zscore_threshold

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        if df.empty:
            return []

        anomalies: list[AnomalyEvent] = []
        window = config.anomaly.rolling_window_hours

        for (region, category), grp in df.groupby(["region", "category"]):
            grp = grp.sort_values("snapshot_ts").reset_index(drop=True)

            for metric in self.METRICS:
                if metric not in grp.columns:
                    continue
                series = grp[metric].astype(float)
                if len(series) < 4:
                    continue

                roll_mean = series.rolling(window, min_periods=3).mean()
                roll_std  = series.rolling(window, min_periods=3).std().replace(0, np.nan)
                z_scores  = (series - roll_mean) / roll_std

                for i, (z, val, mean) in enumerate(
                    zip(z_scores, series, roll_mean)
                ):
                    if pd.isna(z) or abs(z) < self.threshold:
                        continue

                    ts_row = grp.loc[i, "snapshot_ts"]
                    logger.info(
                        "Z-score anomaly: %s/%s %s  z=%.2f  val=%.2f  exp=%.2f",
                        region, category, metric, z, val, mean
                    )
                    anomalies.append(AnomalyEvent(
                        metric_name=metric,
                        metric_value=float(val),
                        expected_value=float(mean) if not pd.isna(mean) else 0.0,
                        z_score=float(z),
                        isolation_score=0.0,   # filled later by IF detector
                        severity=_classify_severity(z),
                        region=str(region),
                        category=str(category),
                        window_start=ts_row - pd.Timedelta(hours=window),
                        window_end=ts_row,
                        raw_context={"method": "zscore", "rolling_window_h": window},
                    ))

        return anomalies


# ── ML: Isolation Forest ──────────────────────────────────────────────────────

class IsolationForestDetector:
    """Multivariate anomaly detection on aggregated feature matrix."""

    FEATURES = ["total_revenue", "avg_order_val", "return_rate",
                "total_orders", "conversion_rt"]

    def __init__(self):
        self.contamination = config.anomaly.isolation_contamination
        self._model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        feat = self._extract_features(df)
        if len(feat) < config.anomaly.min_samples_for_ml:
            logger.warning("Not enough samples for Isolation Forest (%d)", len(feat))
            return
        X = self._scaler.fit_transform(feat)
        self._model.fit(X)
        self._fitted = True
        logger.info("Isolation Forest fitted on %d samples", len(X))

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        if df.empty or not self._fitted:
            return []

        feat = self._extract_features(df)
        if feat.empty:
            return []

        X      = self._scaler.transform(feat)
        labels = self._model.predict(X)           # -1 = anomaly
        scores = self._model.score_samples(X)     # lower = more anomalous

        anomalies = []
        anomaly_rows = df.iloc[feat.index[labels == -1]]

        for idx, row in anomaly_rows.iterrows():
            iso_score = float(scores[feat.index.get_loc(idx)])
            z          = abs(iso_score) * 10         # approximate for severity

            anomalies.append(AnomalyEvent(
                metric_name="multivariate_kpi",
                metric_value=float(row.get("total_revenue", 0)),
                expected_value=0.0,
                z_score=z,
                isolation_score=iso_score,
                severity=_classify_severity(z),
                region=str(row.get("region", "unknown")),
                category=str(row.get("category", "unknown")),
                window_start=row.get("snapshot_ts"),
                window_end=row.get("snapshot_ts"),
                raw_context={"method": "isolation_forest", "score": iso_score},
            ))

        logger.info("Isolation Forest found %d anomalies", len(anomalies))
        return anomalies

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.FEATURES if c in df.columns]
        feat = df[cols].copy().dropna()
        return feat


# ── Ensemble orchestrator ─────────────────────────────────────────────────────

class AnomalyDetectionEngine:
    """Combines Z-score and Isolation Forest; deduplicates and persists results."""

    def __init__(self, engine: Engine):
        self.engine    = engine
        self.zscore    = ZScoreDetector()
        self.isoforest = IsolationForestDetector()

    def run(self) -> list[dict]:
        logger.info("Running anomaly detection cycle…")
        df = _load_kpi_window(self.engine)

        if df.empty:
            logger.warning("No KPI data found. Skipping anomaly detection.")
            return []

        # Fit Isolation Forest on full window, then detect on latest slice
        self.isoforest.fit(df)
        latest_df = df[df["snapshot_ts"] >= df["snapshot_ts"].max() - pd.Timedelta(hours=2)]

        z_anomalies  = self.zscore.detect(df)
        if_anomalies = self.isoforest.detect(latest_df)

        # Merge + deduplicate (same region+category+metric within same window)
        all_anomalies = z_anomalies + if_anomalies
        unique        = self._deduplicate(all_anomalies)

        # Persist
        saved = self._persist(unique)
        logger.info("Anomaly cycle complete: %d detected, %d persisted", len(unique), saved)
        return [self._to_dict(a) for a in unique]

    # ── Dedup & persist ──────────────────────────────────────────────────────

    def _deduplicate(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        seen = set()
        out  = []
        for e in sorted(events, key=lambda x: -abs(x.z_score)):
            key = (e.region, e.category, e.metric_name)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    def _persist(self, events: list[AnomalyEvent]) -> int:
        if not events:
            return 0
        rows = [self._to_dict(e) for e in events]
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO anomaly_events
                    (metric_name, metric_value, expected_value, z_score,
                     isolation_score, severity, region, category,
                     window_start, window_end, raw_context)
                VALUES
                    (:metric_name, :metric_value, :expected_value, :z_score,
                     :isolation_score, :severity, :region, :category,
                     :window_start, :window_end, CAST(:raw_context AS jsonb))
            """), rows)
        return len(rows)

    def _to_dict(self, e: AnomalyEvent) -> dict:
        import json
        return {
            "metric_name":     e.metric_name,
            "metric_value":    round(e.metric_value, 4),
            "expected_value":  round(e.expected_value, 4),
            "z_score":         round(e.z_score, 4),
            "isolation_score": round(e.isolation_score, 6),
            "severity":        e.severity,
            "region":          e.region,
            "category":        e.category,
            "window_start":    e.window_start,
            "window_end":      e.window_end,
            "raw_context":     json.dumps(e.raw_context),
        }

    def fetch_latest_anomalies(self, limit: int = 10) -> list[dict]:
        sql = f"""
        SELECT anomaly_id, metric_name, metric_value, expected_value,
               z_score, severity, region, category, detected_at
        FROM anomaly_events
        ORDER BY detected_at DESC
        LIMIT {limit}
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df.to_dict("records")
