"""
tests/test_causal_agents.py
Tests for causal inference and multi-agent orchestration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from causal.inference import _regression_fallback, CausalInferenceEngine
from agents.orchestrator import _extract_priority, _extract_action, _mock_llm_response


# ── Causal regression fallback ────────────────────────────────────────────────

class TestRegressionFallback:

    def _make_df(self, n=100, slope=5.0, noise=1.0):
        rng = np.random.default_rng(7)
        x   = rng.uniform(0, 20, n)
        y   = slope * x + rng.normal(0, noise, n)
        return pd.DataFrame({"treatment": x, "outcome": y})

    def test_positive_ate_detected(self):
        df     = self._make_df(slope=5.0)
        result = _regression_fallback(df, "treatment", "outcome")
        assert result["ate"] > 0
        assert result["method"] == "ols_bootstrap"

    def test_negative_ate_detected(self):
        df     = self._make_df(slope=-3.0)
        result = _regression_fallback(df, "treatment", "outcome")
        assert result["ate"] < 0

    def test_confidence_in_range(self):
        df     = self._make_df(slope=5.0)
        result = _regression_fallback(df, "treatment", "outcome")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_insufficient_data(self):
        df     = pd.DataFrame({"treatment": [1, 2], "outcome": [3, 4]})
        result = _regression_fallback(df, "treatment", "outcome")
        assert result["method"] == "insufficient_data"
        assert result["ate"] == 0.0

    def test_strong_signal_high_confidence(self):
        """Perfect linear relationship → very low p-value."""
        x  = np.linspace(0, 100, 200)
        df = pd.DataFrame({"treatment": x, "outcome": 3.0 * x})
        r  = _regression_fallback(df, "treatment", "outcome")
        assert r["p_value"] < 0.05


class TestCausalInferenceEngine:

    def _mock_engine_with_df(self, df: pd.DataFrame) -> MagicMock:
        m = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__  = MagicMock(return_value=False)
        m.connect.return_value = ctx
        return m

    def test_returns_dict_with_required_keys(self):
        engine  = self._mock_engine_with_df(pd.DataFrame())
        causal  = CausalInferenceEngine(engine)
        anomaly = {"region": "North", "category": "Electronics", "metric_name": "total_revenue"}

        # Patch DB calls to return empty → should hit _empty_result
        with patch("causal.inference._load_order_features", return_value=pd.DataFrame()):
            with patch("causal.inference._load_event_features", return_value=pd.DataFrame()):
                result = causal.analyze(anomaly)

        required = {"cause_variable", "effect_variable", "ate", "confidence",
                    "method", "explanation", "supporting_data"}
        assert required.issubset(result.keys())

    def test_returns_best_causal_pair(self):
        rng = np.random.default_rng(42)
        n   = 80
        x   = rng.uniform(0, 30, n)
        df  = pd.DataFrame({
            "hour_bucket":  pd.date_range("2024-01-01", periods=n, freq="h"),
            "avg_discount": x,
            "revenue":      500 + 120 * x + rng.normal(0, 50, n),
            "return_rate":  0.05 + 0.002 * x,
            "avg_price":    rng.uniform(50, 200, n),
            "order_count":  rng.integers(50, 200, n),
            "avg_qty":      rng.uniform(1, 5, n),
        })
        engine = MagicMock()
        causal = CausalInferenceEngine(engine)

        with patch("causal.inference._load_order_features", return_value=df):
            with patch("causal.inference._load_event_features", return_value=pd.DataFrame()):
                result = causal.analyze({"region": "N", "category": "E",
                                         "metric_name": "total_revenue"})

        assert result["confidence"] > 0.5
        assert result["ate"] != 0.0


# ── Agent helpers ─────────────────────────────────────────────────────────────

class TestAgentHelpers:

    def test_extract_priority_p0(self):
        assert _extract_priority("This is P0 critical") == "P0"

    def test_extract_priority_p1(self):
        assert _extract_priority("DECISION [P1]: freeze discounts") == "P1"

    def test_extract_priority_default(self):
        assert _extract_priority("No priority mentioned") == "P2"

    def test_extract_priority_highest_wins(self):
        # First match wins
        assert _extract_priority("P0 and P1 mentioned") == "P0"

    def test_extract_action_numbered(self):
        text   = "DECISION:\n1) Freeze promotions immediately\n2) Alert team"
        action = _extract_action(text)
        assert "Freeze" in action

    def test_extract_action_fallback(self):
        text   = "Take action now for the good of the system"
        action = _extract_action(text)
        assert len(action) > 0

    def test_mock_llm_analyst(self):
        resp = _mock_llm_response("ANALYST AGENT", [{"role": "user", "content": "anomaly data"}])
        assert "ANALYST" in resp

    def test_mock_llm_causal(self):
        resp = _mock_llm_response("CAUSAL AGENT", [{"role": "user", "content": "cause this"}])
        assert "CAUSAL" in resp

    def test_mock_llm_decision(self):
        resp = _mock_llm_response("DECISION AGENT", [{"role": "user", "content": "decide"}])
        assert "DECISION" in resp


class TestAgentOrchestrator:

    def _make_anomaly(self) -> dict:
        return {
            "anomaly_id": 99,
            "metric_name": "total_revenue",
            "metric_value": 85000.0,
            "expected_value": 12000.0,
            "z_score": 7.5,
            "severity": "HIGH",
            "region": "East",
            "category": "Sports",
            "detected_at": "2024-01-01T12:00:00Z",
        }

    def _make_finding(self) -> dict:
        return {
            "cause_variable":  "avg_discount",
            "effect_variable": "total_revenue",
            "ate":             88.5,
            "confidence":      0.82,
            "method":          "ols_bootstrap",
            "explanation":     "avg_discount caused total_revenue with ATE=88.5",
            "hypothesis":      "High discount → revenue anomaly",
            "supporting_data": {},
        }

    def test_run_produces_result(self):
        from agents.orchestrator import AgentOrchestrator

        mock_engine = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__  = MagicMock(return_value=False)
        mock_engine.begin.return_value = ctx

        orch   = AgentOrchestrator(mock_engine)
        result = orch.run(self._make_anomaly(), self._make_finding())

        assert result.priority in ("P0", "P1", "P2", "P3")
        assert len(result.analyst_output) > 10
        assert len(result.causal_output) > 10
        assert len(result.decision_output) > 10
        assert result.anomaly_id == 99

    def test_debate_history_populated(self):
        from agents.orchestrator import AgentOrchestrator

        mock_engine = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__  = MagicMock(return_value=False)
        mock_engine.begin.return_value = ctx

        orch   = AgentOrchestrator(mock_engine)
        result = orch.run(self._make_anomaly(), self._make_finding())

        assert len(result.debate_history) >= 3   # Analyst, Causal, Decision
        agents = {r["agent"] for r in result.debate_history}
        assert agents >= {"Analyst", "Causal", "Decision"}
