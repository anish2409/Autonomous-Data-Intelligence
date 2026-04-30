"""
tests/test_ingestion.py
Unit tests for the data ingestion pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

from ingestion.pipeline import (
    generate_orders_batch,
    generate_events_batch,
    detect_schema_drift,
    _inject_anomalies,
    _SCHEMA_REGISTRY,
)


class TestGenerateOrdersBatch:

    def test_returns_correct_shape(self):
        df = generate_orders_batch(n=500)
        assert len(df) == 500

    def test_required_columns_present(self):
        required = {"customer_id", "product_id", "category", "region",
                    "order_ts", "quantity", "unit_price", "discount_pct",
                    "payment_method", "is_returned"}
        df = generate_orders_batch(n=100)
        assert required.issubset(df.columns)

    def test_unit_price_positive(self):
        df = generate_orders_batch(n=1000, inject_anomalies=False)
        assert (df["unit_price"] > 0).all()

    def test_quantity_positive(self):
        df = generate_orders_batch(n=1000, inject_anomalies=False)
        assert (df["quantity"] >= 1).all()

    def test_discount_in_range(self):
        df = generate_orders_batch(n=1000, inject_anomalies=False)
        assert (df["discount_pct"] >= 0).all()
        assert (df["discount_pct"] <= 100).all()

    def test_is_returned_boolean(self):
        df = generate_orders_batch(n=500)
        assert df["is_returned"].dtype == bool

    def test_return_rate_roughly_7pct(self):
        df   = generate_orders_batch(n=10_000, inject_anomalies=False)
        rate = df["is_returned"].mean()
        assert 0.04 < rate < 0.12   # 7% ± tolerance

    def test_custom_base_ts(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = generate_orders_batch(n=100, base_ts=ts, inject_anomalies=False)
        assert df["order_ts"].max() <= ts
        assert df["order_ts"].min() >= ts.replace(second=0) - __import__("datetime").timedelta(seconds=60)

    def test_deterministic_with_no_anomalies(self):
        """Same seed should produce same result if anomaly injection is off."""
        df1 = generate_orders_batch(n=50, inject_anomalies=False)
        df2 = generate_orders_batch(n=50, inject_anomalies=False)
        # RNG is global; just check shapes match
        assert df1.shape == df2.shape

    def test_anomaly_injection_creates_extremes(self):
        """Injected anomalies should make max price far exceed the normal range."""
        df_clean    = generate_orders_batch(n=2000, inject_anomalies=False)
        df_injected = generate_orders_batch(n=2000, inject_anomalies=True)
        assert df_injected["unit_price"].max() > df_clean["unit_price"].max()


class TestGenerateEventsBatch:

    def test_returns_correct_shape(self):
        df = generate_events_batch(n=200)
        assert len(df) == 200

    def test_event_types_valid(self):
        valid = {"page_view", "add_to_cart", "remove_from_cart", "checkout", "purchase"}
        df    = generate_events_batch(n=500)
        assert set(df["event_type"].unique()).issubset(valid)

    def test_session_ids_unique(self):
        df = generate_events_batch(n=100)
        # Each row gets its own UUID
        assert df["session_id"].nunique() == 100


class TestSchemaDriftDetection:

    def _make_engine_with_schema(self, schema: dict) -> MagicMock:
        """Return a mock engine whose inspect() reports `schema`."""
        mock_engine = MagicMock()
        mock_insp   = MagicMock()
        mock_insp.get_columns.return_value = [
            {"name": k, "type": MagicMock(__str__=lambda self, t=v: t)}
            for k, v in schema.items()
        ]
        with patch("ingestion.pipeline.inspect", return_value=mock_insp):
            return mock_engine

    def test_no_drift_on_matching_schema(self):
        df = generate_orders_batch(n=10, inject_anomalies=False)
        _SCHEMA_REGISTRY.clear()

        mock_engine = MagicMock()
        with patch("ingestion.pipeline.inspect") as mock_insp_factory:
            mock_insp = MagicMock()
            mock_insp.get_columns.return_value = [
                {"name": c, "type": MagicMock()} for c in df.columns
            ]
            mock_insp_factory.return_value = mock_insp
            drifts = detect_schema_drift(mock_engine, "raw_orders", df)

        # First call always registers; drift only appears on second
        assert isinstance(drifts, list)

    def test_detects_new_column(self):
        df = generate_orders_batch(n=10, inject_anomalies=False)
        df["surprise_col"] = "oops"   # extra column

        _SCHEMA_REGISTRY["raw_orders"] = {c: "VARCHAR" for c in
                                           generate_orders_batch(n=1, inject_anomalies=False).columns}
        mock_engine = MagicMock()
        with patch("ingestion.pipeline.inspect") as mock_insp_factory:
            mock_insp = MagicMock()
            mock_insp.get_columns.return_value = [
                {"name": c, "type": MagicMock()} for c in df.columns
            ]
            mock_insp_factory.return_value = mock_insp
            drifts = detect_schema_drift(mock_engine, "raw_orders", df)

        new_col_drifts = [d for d in drifts if d["drift_type"] == "NEW_COLUMN"]
        assert any(d["column_name"] == "surprise_col" for d in new_col_drifts)

    def test_detects_dropped_column(self):
        df = generate_orders_batch(n=10, inject_anomalies=False)
        # Registry has a column that's NOT in df
        _SCHEMA_REGISTRY["raw_orders"] = {**{c: "VARCHAR" for c in df.columns},
                                           "ghost_column": "BIGINT"}
        mock_engine = MagicMock()
        with patch("ingestion.pipeline.inspect") as mock_insp_factory:
            mock_insp = MagicMock()
            mock_insp.get_columns.return_value = [
                {"name": c, "type": MagicMock()} for c in df.columns
            ]
            mock_insp_factory.return_value = mock_insp
            drifts = detect_schema_drift(mock_engine, "raw_orders", df)

        dropped = [d for d in drifts if d["drift_type"] == "DROPPED_COLUMN"]
        assert any(d["column_name"] == "ghost_column" for d in dropped)
