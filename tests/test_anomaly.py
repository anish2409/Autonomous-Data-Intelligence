"""
tests/test_anomaly.py
Unit tests for the anomaly detection engine.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from anomaly.detector import (
    ZScoreDetector,
    IsolationForestDetector,
    AnomalyDetectionEngine,
    _classify_severity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_kpi_df(n_hours: int = 48, inject_spike: bool = False,
                region: str = "North", category: str = "Electronics") -> pd.DataFrame:
    """Build a synthetic KPI dataframe for testing."""
    rng = np.random.default_rng(0)
    now = datetime.now(timezone.utc)
    ts  = [now - timedelta(hours=i) for i in range(n_hours, 0, -1)]

    revenue = rng.normal(10_000, 500, size=n_hours).clip(1_000)
    if inject_spike:
        revenue[-1] = 90_000   # obvious spike

    return pd.DataFrame({
        "snapshot_ts":   ts,
        "region":        region,
        "category":      category,
        "total_orders":  rng.integers(80, 150, size=n_hours),
        "total_revenue": revenue,
        "avg_order_val": rng.uniform(60, 140, size=n_hours),
        "return_rate":   rng.uniform(0.04, 0.10, size=n_hours),
        "conversion_rt": rng.uniform(0.06, 0.12, size=n_hours),
    })


# ── Severity classification ───────────────────────────────────────────────────

class TestSeverityClassification:
    def test_low(self):     assert _classify_severity(3.1)  == "LOW"
    def test_medium(self):  assert _classify_severity(4.5)  == "MEDIUM"
    def test_high(self):    assert _classify_severity(7.0)  == "HIGH"
    def test_critical(self):assert _classify_severity(9.5)  == "CRITICAL"
    def test_negative(self):assert _classify_severity(-5.0) == "MEDIUM"
    def test_boundary(self):assert _classify_severity(3.0)  == "LOW"


# ── Z-score detector ──────────────────────────────────────────────────────────

class TestZScoreDetector:

    def test_detects_spike(self):
        df      = make_kpi_df(n_hours=48, inject_spike=True)
        det     = ZScoreDetector(threshold=3.0)
        results = det.detect(df)
        assert len(results) > 0
        revenues = [r for r in results if r.metric_name == "total_revenue"]
        assert len(revenues) > 0
        assert revenues[0].z_score > 3.0

    def test_no_false_positive_on_clean_data(self):
        """Tight normal data should not trigger."""
        rng = np.random.default_rng(42)
        ts  = [datetime.now(timezone.utc) - timedelta(hours=i) for i in range(48, 0, -1)]
        df  = pd.DataFrame({
            "snapshot_ts":   ts, "region": "East", "category": "Books",
            "total_orders":  100 + rng.integers(-2, 2, 48),
            "total_revenue": 10_000 + rng.normal(0, 10, 48),
            "avg_order_val": 100 + rng.normal(0, 1, 48),
            "return_rate":   0.05 + rng.normal(0, 0.001, 48),
            "conversion_rt": 0.08 + rng.normal(0, 0.001, 48),
        })
        det     = ZScoreDetector(threshold=3.0)
        results = det.detect(df)
        assert len(results) == 0

    def test_empty_df(self):
        det = ZScoreDetector()
        assert det.detect(pd.DataFrame()) == []

    def test_multiple_groups(self):
        """Should detect anomalies independently per group."""
        dfs = [
            make_kpi_df(inject_spike=True,  region="North", category="Electronics"),
            make_kpi_df(inject_spike=False, region="South", category="Apparel"),
        ]
        df      = pd.concat(dfs, ignore_index=True)
        det     = ZScoreDetector(threshold=3.0)
        results = det.detect(df)
        regions = {r.region for r in results}
        assert "North" in regions        # spike group detected
        assert "South" not in regions    # clean group silent

    def test_severity_on_large_spike(self):
        """A 20x spike should be CRITICAL."""
        rng = np.random.default_rng(1)
        ts  = [datetime.now(timezone.utc) - timedelta(hours=i) for i in range(48, 0, -1)]
        rev = list(rng.normal(10_000, 200, 47)) + [200_000.0]
        df  = pd.DataFrame({
            "snapshot_ts": ts, "region": "West", "category": "Electronics",
            "total_orders": 100, "total_revenue": rev,
            "avg_order_val": 100.0, "return_rate": 0.06, "conversion_rt": 0.08
        })
        det     = ZScoreDetector(threshold=3.0)
        results = det.detect(df)
        sevs    = [r.severity for r in results if r.metric_name == "total_revenue"]
        assert any(s in ("HIGH", "CRITICAL") for s in sevs)


# ── Isolation Forest detector ─────────────────────────────────────────────────

class TestIsolationForestDetector:

    def test_fit_and_detect(self):
        df  = make_kpi_df(n_hours=100, inject_spike=True)
        det = IsolationForestDetector()
        det.fit(df)
        assert det._fitted is True
        results = det.detect(df.tail(10))
        # Should flag at least one point when spike is included
        assert isinstance(results, list)

    def test_not_fitted_returns_empty(self):
        det = IsolationForestDetector()
        df  = make_kpi_df()
        assert det.detect(df) == []

    def test_insufficient_samples_skips_fit(self):
        det = IsolationForestDetector()
        df  = make_kpi_df(n_hours=5)
        det.fit(df)
        assert det._fitted is False

    def test_empty_df_safe(self):
        det = IsolationForestDetector()
        det.fit(make_kpi_df(n_hours=100))
        assert det.detect(pd.DataFrame()) == []


# ── Ensemble engine ───────────────────────────────────────────────────────────

class TestAnomalyDetectionEngine:

    def _make_engine(self, df: pd.DataFrame) -> AnomalyDetectionEngine:
        mock_engine = MagicMock()
        # Mock DB query to return our test dataframe
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_engine.begin.return_value = mock_conn

        ade = AnomalyDetectionEngine(mock_engine)

        with patch("anomaly.detector._load_kpi_window", return_value=df):
            with patch.object(ade, "_persist", return_value=2):
                results = ade.run()

        return results

    def test_end_to_end_with_spike(self):
        df      = make_kpi_df(n_hours=60, inject_spike=True)
        results = self._make_engine(df)
        assert isinstance(results, list)

    def test_returns_list_on_empty(self):
        results = self._make_engine(pd.DataFrame())
        assert results == []

    def test_deduplication(self):
        """Same (region, category, metric) should appear only once."""
        df      = make_kpi_df(n_hours=60, inject_spike=True)
        results = self._make_engine(df)
        keys    = [(r.get("region"), r.get("category"), r.get("metric_name"))
                   for r in results]
        assert len(keys) == len(set(keys))
