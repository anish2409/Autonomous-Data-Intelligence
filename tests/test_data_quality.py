"""
tests/test_data_quality.py
Data quality validators — run after ingestion as automated guardrails.
These are great-expectation-style checks implemented without the GE dependency.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Callable


# ── Expectation framework (mini Great Expectations) ──────────────────────────

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    detail:  str
    failing_rows: int = 0


@dataclass
class DataQualityReport:
    table: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self): return all(c.passed for c in self.checks)

    @property
    def pass_rate(self): return sum(c.passed for c in self.checks) / max(len(self.checks), 1)

    def summary(self) -> str:
        lines = [f"Data Quality Report: {self.table}",
                 f"  Pass rate: {self.pass_rate:.0%}  ({sum(c.passed for c in self.checks)}/{len(self.checks)})"]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} {c.name}: {c.detail}")
        return "\n".join(lines)


def expect_no_nulls(df: pd.DataFrame, col: str) -> CheckResult:
    n = df[col].isna().sum()
    return CheckResult(f"no_nulls:{col}", n == 0,
                       f"{n} null rows" if n else "clean", failing_rows=int(n))


def expect_values_in_set(df: pd.DataFrame, col: str, valid: set) -> CheckResult:
    bad = ~df[col].isin(valid)
    n   = bad.sum()
    return CheckResult(f"values_in_set:{col}", n == 0,
                       f"{n} invalid values" if n else f"all in {valid}", failing_rows=int(n))


def expect_column_between(df: pd.DataFrame, col: str, lo: float, hi: float) -> CheckResult:
    bad = ~df[col].between(lo, hi)
    n   = bad.sum()
    return CheckResult(f"between:[{lo},{hi}]:{col}", n == 0,
                       f"{n} out-of-range values" if n else f"all in [{lo}, {hi}]",
                       failing_rows=int(n))


def expect_unique(df: pd.DataFrame, col: str) -> CheckResult:
    dups = df[col].duplicated().sum()
    return CheckResult(f"unique:{col}", dups == 0,
                       f"{dups} duplicates" if dups else "unique", failing_rows=int(dups))


def expect_row_count_between(df: pd.DataFrame, lo: int, hi: int) -> CheckResult:
    n = len(df)
    return CheckResult("row_count", lo <= n <= hi,
                       f"{n} rows (expected {lo}–{hi})")


def expect_mean_between(df: pd.DataFrame, col: str, lo: float, hi: float) -> CheckResult:
    m = float(df[col].mean())
    return CheckResult(f"mean_between:[{lo},{hi}]:{col}", lo <= m <= hi,
                       f"mean={m:.4f}")


def expect_no_future_timestamps(df: pd.DataFrame, col: str) -> CheckResult:
    import datetime
    now = pd.Timestamp.now(tz="UTC")
    # Normalise to UTC
    ts  = pd.to_datetime(df[col], utc=True)
    n   = (ts > now + pd.Timedelta(minutes=5)).sum()
    return CheckResult(f"no_future_ts:{col}", n == 0,
                       f"{n} future timestamps" if n else "clean",
                       failing_rows=int(n))


# ── Table-specific validators ─────────────────────────────────────────────────

CATEGORIES = {"Electronics", "Apparel", "Home & Kitchen", "Sports", "Books", "Beauty"}
REGIONS    = {"North", "South", "East", "West", "International"}
PAYMENTS   = {"credit_card", "debit_card", "paypal", "crypto", "bank_transfer"}


def validate_raw_orders(df: pd.DataFrame) -> DataQualityReport:
    report = DataQualityReport("raw_orders")
    checks = [
        expect_row_count_between(df, 1, 10_000_000),
        expect_no_nulls(df, "customer_id"),
        expect_no_nulls(df, "order_ts"),
        expect_no_nulls(df, "unit_price"),
        expect_column_between(df, "unit_price", 0.01, 1_000_000),
        expect_column_between(df, "quantity", 1, 10_000),
        expect_column_between(df, "discount_pct", 0, 100),
        expect_values_in_set(df, "category", CATEGORIES),
        expect_values_in_set(df, "region", REGIONS),
        expect_values_in_set(df, "payment_method", PAYMENTS),
        expect_mean_between(df, "unit_price", 5, 1_000),
        expect_no_future_timestamps(df, "order_ts"),
    ]
    report.checks = checks
    return report


def validate_kpi_snapshots(df: pd.DataFrame) -> DataQualityReport:
    report = DataQualityReport("kpi_snapshots")
    checks = [
        expect_row_count_between(df, 1, 5_000_000),
        expect_no_nulls(df, "snapshot_ts"),
        expect_no_nulls(df, "total_revenue"),
        expect_column_between(df, "return_rate", 0, 1),
        expect_column_between(df, "total_revenue", 0, 1e9),
        expect_values_in_set(df, "period", {"hourly", "daily"}),
        expect_no_future_timestamps(df, "snapshot_ts"),
    ]
    report.checks = checks
    return report


def validate_anomaly_events(df: pd.DataFrame) -> DataQualityReport:
    report = DataQualityReport("anomaly_events")
    checks = [
        expect_no_nulls(df, "metric_name"),
        expect_no_nulls(df, "severity"),
        expect_values_in_set(df, "severity", {"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
        expect_column_between(df, "z_score", -100, 100),
    ]
    report.checks = checks
    return report


# ── Pytest integration ────────────────────────────────────────────────────────

class TestOrdersDataQuality:
    """Run DQ checks against synthetic order data (mirrors prod validation)."""

    def _clean_df(self, n=1000) -> pd.DataFrame:
        from ingestion.pipeline import generate_orders_batch
        return generate_orders_batch(n=n, inject_anomalies=False)

    def test_no_null_customer_ids(self):
        df     = self._clean_df()
        result = expect_no_nulls(df, "customer_id")
        assert result.passed, result.detail

    def test_no_null_order_ts(self):
        df     = self._clean_df()
        result = expect_no_nulls(df, "order_ts")
        assert result.passed, result.detail

    def test_unit_price_positive(self):
        df     = self._clean_df()
        result = expect_column_between(df, "unit_price", 0.01, 1_000_000)
        assert result.passed, result.detail

    def test_discount_in_valid_range(self):
        df     = self._clean_df()
        result = expect_column_between(df, "discount_pct", 0, 100)
        assert result.passed, result.detail

    def test_categories_valid(self):
        df     = self._clean_df()
        result = expect_values_in_set(df, "category", CATEGORIES)
        assert result.passed, result.detail

    def test_regions_valid(self):
        df     = self._clean_df()
        result = expect_values_in_set(df, "region", REGIONS)
        assert result.passed, result.detail

    def test_payment_methods_valid(self):
        df     = self._clean_df()
        result = expect_values_in_set(df, "payment_method", PAYMENTS)
        assert result.passed, result.detail

    def test_no_future_timestamps(self):
        df     = self._clean_df()
        result = expect_no_future_timestamps(df, "order_ts")
        assert result.passed, result.detail

    def test_full_report_passes(self):
        df     = self._clean_df(n=2000)
        report = validate_raw_orders(df)
        print("\n" + report.summary())
        assert report.pass_rate >= 0.90, f"DQ pass rate below 90%:\n{report.summary()}"


class TestKpiSnapshotDQ:

    def _make_kpi_df(self) -> pd.DataFrame:
        import datetime
        rng = np.random.default_rng(0)
        n   = 200
        return pd.DataFrame({
            "snapshot_ts":   pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "period":        "hourly",
            "region":        rng.choice(list(REGIONS), n),
            "category":      rng.choice(list(CATEGORIES), n),
            "total_orders":  rng.integers(50, 300, n),
            "total_revenue": rng.uniform(5_000, 50_000, n),
            "avg_order_val": rng.uniform(50, 300, n),
            "return_rate":   rng.uniform(0.02, 0.15, n),
            "conversion_rt": rng.uniform(0.05, 0.15, n),
        })

    def test_return_rate_in_range(self):
        df     = self._make_kpi_df()
        result = expect_column_between(df, "return_rate", 0, 1)
        assert result.passed

    def test_full_report(self):
        df     = self._make_kpi_df()
        report = validate_kpi_snapshots(df)
        print("\n" + report.summary())
        assert report.pass_rate == 1.0


class TestAnomalyEventsDQ:

    def _make_anomaly_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        n   = 50
        return pd.DataFrame({
            "metric_name": rng.choice(["total_revenue", "return_rate"], n),
            "severity":    rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], n),
            "z_score":     rng.uniform(3, 12, n),
            "region":      rng.choice(list(REGIONS), n),
            "category":    rng.choice(list(CATEGORIES), n),
        })

    def test_severities_valid(self):
        df     = self._make_anomaly_df()
        result = expect_values_in_set(df, "severity", {"LOW","MEDIUM","HIGH","CRITICAL"})
        assert result.passed

    def test_z_scores_bounded(self):
        df     = self._make_anomaly_df()
        result = expect_column_between(df, "z_score", -100, 100)
        assert result.passed
