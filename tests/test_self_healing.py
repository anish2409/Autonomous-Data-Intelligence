"""
tests/test_self_healing.py
Tests for the self-healing pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, call
from self_healing.healer import (
    DriftEvent,
    _build_heal_query,
    _heal_dropped_column,
    _heal_new_column,
    _heal_null_violation,
    rewrite_dbt_model,
    SelfHealingPipeline,
)


class TestHealQueryGeneration:

    def test_dropped_bigint_column(self):
        drift = DriftEvent("raw_orders", "customer_id", "DROPPED_COLUMN", "BIGINT", None)
        q = _heal_dropped_column(drift)
        assert "ADD COLUMN IF NOT EXISTS" in q
        assert "customer_id" in q
        assert "BIGINT" in q

    def test_dropped_varchar_column(self):
        drift = DriftEvent("raw_orders", "region", "DROPPED_COLUMN", "VARCHAR", None)
        q = _heal_dropped_column(drift)
        assert "VARCHAR(128)" in q

    def test_dropped_boolean_column(self):
        drift = DriftEvent("raw_orders", "is_returned", "DROPPED_COLUMN", "BOOLEAN", None)
        q = _heal_dropped_column(drift)
        assert "BOOLEAN" in q

    def test_new_column_adds_comment(self):
        drift = DriftEvent("raw_orders", "mystery_col", "NEW_COLUMN", None, "TEXT")
        q = _heal_new_column(drift)
        assert "COMMENT ON COLUMN" in q
        assert "mystery_col" in q
        assert "AUTO_DETECTED" in q

    def test_null_violation_numeric_fill(self):
        drift = DriftEvent("raw_orders", "unit_price", "NULL_VIOLATION",
                           "NOT NULL", "5 NULL rows found")
        q = _heal_null_violation(drift)
        assert "UPDATE raw_orders" in q
        assert "unit_price" in q
        assert "IS NULL" in q

    def test_null_violation_timestamp_fill(self):
        drift = DriftEvent("raw_orders", "order_ts", "NULL_VIOLATION",
                           "NOT NULL", "2 NULL rows found")
        q = _heal_null_violation(drift)
        assert "1970-01-01" in q

    def test_type_change_returns_none(self):
        drift = DriftEvent("raw_orders", "quantity", "TYPE_CHANGE",
                           "INTEGER", "VARCHAR")
        result = _build_heal_query(drift)
        assert result is None

    def test_build_heal_query_routes_correctly(self):
        drift_new  = DriftEvent("t", "c", "NEW_COLUMN",     None,   "TEXT")
        drift_drop = DriftEvent("t", "c", "DROPPED_COLUMN", "INT",  None)
        drift_null = DriftEvent("raw_orders", "unit_price", "NULL_VIOLATION",
                                "NOT NULL", "1 NULL row")

        assert _build_heal_query(drift_new)  is not None
        assert _build_heal_query(drift_drop) is not None
        assert _build_heal_query(drift_null) is not None


class TestDbtModelRewriter:

    def test_creates_file(self, tmp_path):
        drifts = [
            DriftEvent("raw_orders", "new_col", "NEW_COLUMN", None, "TEXT"),
        ]
        path = rewrite_dbt_model(drifts, output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_file_content_includes_drift_summary(self, tmp_path):
        drifts = [
            DriftEvent("raw_orders", "extra", "NEW_COLUMN", None, "TEXT"),
            DriftEvent("raw_orders", "gone",  "DROPPED_COLUMN", "BIGINT", None),
        ]
        path = rewrite_dbt_model(drifts, output_dir=str(tmp_path))
        content = open(path).read()
        assert "NEW_COLUMN" in content
        assert "DROPPED_COLUMN" in content
        assert "AUTO-GENERATED" in content

    def test_null_violation_adds_filter(self, tmp_path):
        drifts = [
            DriftEvent("raw_orders", "unit_price", "NULL_VIOLATION",
                       "NOT NULL", "3 rows"),
        ]
        path    = rewrite_dbt_model(drifts, output_dir=str(tmp_path))
        content = open(path).read()
        assert "unit_price IS NOT NULL" in content


class TestSelfHealingPipeline:

    def _make_engine(self) -> MagicMock:
        m   = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__  = MagicMock(return_value=False)
        m.begin.return_value   = ctx
        m.connect.return_value = ctx
        return m

    def test_run_returns_empty_when_no_drift(self):
        engine = self._make_engine()
        healer = SelfHealingPipeline(engine)
        with patch("self_healing.healer._detect_drift",          return_value=[]):
            with patch("self_healing.healer._detect_null_violations", return_value=[]):
                results = healer.run()
        assert results == []

    def test_run_heals_dropped_column(self):
        engine = self._make_engine()
        healer = SelfHealingPipeline(engine)
        drift  = DriftEvent("raw_orders", "customer_id", "DROPPED_COLUMN", "BIGINT", None)

        with patch("self_healing.healer._detect_drift", return_value=[drift]):
            with patch("self_healing.healer._detect_null_violations", return_value=[]):
                with patch("self_healing.healer._validate_and_apply",
                           return_value=MagicMock(healed=True, validated=True,
                                                  query="ALTER TABLE …", error=None,
                                                  drift=drift)) as mock_apply:
                    with patch("self_healing.healer._log_drift"):
                        results = healer.run()

        assert len(results) == 1
        mock_apply.assert_called_once()

    def test_health_report_returns_dict(self):
        import pandas as pd
        engine = self._make_engine()
        healer = SelfHealingPipeline(engine)
        mock_df = pd.DataFrame([
            {"drift_type": "NEW_COLUMN", "count": 3, "healed": 2},
            {"drift_type": "NULL_VIOLATION", "count": 1, "healed": 1},
        ])
        with patch("self_healing.healer.pd.read_sql", return_value=mock_df):
            report = healer.health_report()

        assert "total_drifts" in report
        assert "total_healed" in report
        assert "breakdown" in report
        assert report["total_drifts"] == 4
        assert report["total_healed"] == 3
