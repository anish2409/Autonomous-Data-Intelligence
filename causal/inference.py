"""
causal/inference.py
Causal analysis engine.
Uses DoWhy (with econml / linear regression fallback) to estimate
Average Treatment Effects and identify root-cause variables for each anomaly.
"""
import json
import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("adi.causal")

# DoWhy is optional; fall back gracefully
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not installed; using regression-based causal fallback")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_order_features(engine: Engine, region: str, category: str,
                         window_hours: int = 48) -> pd.DataFrame:
    """Load raw order features for causal analysis."""
    sql = """
    SELECT
        date_trunc('hour', order_ts)                AS hour_bucket,
        COUNT(*)                                     AS order_count,
        AVG(unit_price)                              AS avg_price,
        AVG(discount_pct)                            AS avg_discount,
        AVG(is_returned::int)                        AS return_rate,
        SUM(quantity * unit_price * (1 - discount_pct/100)) AS revenue,
        AVG(quantity)                                AS avg_qty
    FROM raw_orders
    WHERE region   = :region
      AND category = :category
      AND order_ts >= NOW() - INTERVAL ':w hours'
    GROUP BY 1
    ORDER BY 1
    """
    # SQLAlchemy doesn't interpolate INTERVAL cleanly, so use format
    sql_fmt = sql.replace(":w", str(window_hours))
    with engine.connect() as conn:
        df = pd.read_sql(
            text(sql_fmt), conn,
            params={"region": region, "category": category},
            parse_dates=["hour_bucket"]
        )
    return df


def _load_event_features(engine: Engine, window_hours: int = 48) -> pd.DataFrame:
    """Load event-level conversion signals."""
    sql = f"""
    SELECT
        date_trunc('hour', event_ts)   AS hour_bucket,
        event_type,
        COUNT(*)                        AS event_count
    FROM raw_events
    WHERE event_ts >= NOW() - INTERVAL '{window_hours} hours'
    GROUP BY 1, 2
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, parse_dates=["hour_bucket"])
    pivot = df.pivot_table(
        index="hour_bucket", columns="event_type",
        values="event_count", aggfunc="sum", fill_value=0
    )
    pivot.columns = [f"evt_{c}" for c in pivot.columns]
    return pivot.reset_index()


# ── DoWhy wrapper ─────────────────────────────────────────────────────────────

def _dowhy_estimate(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    """Run a DoWhy causal estimate using backdoor linear regression."""
    try:
        confounders = [c for c in df.columns if c not in [treatment, outcome, "hour_bucket"]]
        gml = f"""
        graph [directed 1
            node [ id "{treatment}" label "{treatment}" ]
            node [ id "{outcome}"  label "{outcome}"  ]
        """
        for c in confounders[:3]:   # limit graph complexity
            gml += f'\n  node [ id "{c}" label "{c}" ]\n'
            gml += f'  edge [ source "{c}" target "{treatment}" ]\n'
            gml += f'  edge [ source "{c}" target "{outcome}" ]\n'
        gml += f'  edge [ source "{treatment}" target "{outcome}" ]\n]'

        model = CausalModel(
            data=df.dropna(),
            treatment=treatment,
            outcome=outcome,
            graph=gml,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate   = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        refute     = model.refute_estimate(
            identified, estimate,
            method_name="random_common_cause",
        )
        return {
            "ate":         float(estimate.value),
            "p_value":     float(getattr(refute, "new_effect", 0)),
            "method":      "dowhy_backdoor_lr",
            "refutation":  str(refute)[:200],
        }
    except Exception as exc:
        logger.warning("DoWhy estimation failed: %s", exc)
        return _regression_fallback(df, treatment, outcome)


def _regression_fallback(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    """Ordinary least squares fallback when DoWhy is unavailable or fails."""
    from sklearn.linear_model import LinearRegression
    sub = df[[treatment, outcome]].dropna()
    if len(sub) < 5:
        return {"ate": 0.0, "p_value": 1.0, "method": "insufficient_data", "refutation": ""}
    X = sub[[treatment]].values
    y = sub[outcome].values
    lr = LinearRegression().fit(X, y)
    ate = float(lr.coef_[0])

    # Approximate confidence via bootstrap
    boots = [
        LinearRegression().fit(X[idx], y[idx]).coef_[0]
        for idx in (
            np.random.default_rng(i).choice(len(X), len(X))
            for i in range(200)
        )
    ]
    se = float(np.std(boots))
    p  = float(2 * (1 - min(abs(ate / (se + 1e-9)) / 2, 0.999)))
    return {
        "ate": ate,
        "p_value": p,
        "method": "ols_bootstrap",
        "refutation": f"Bootstrap SE={se:.4f}",
    }


# ── Candidate causal pairs ────────────────────────────────────────────────────

CAUSAL_PAIRS = [
    ("avg_discount",  "revenue",     "High discount → revenue anomaly"),
    ("avg_discount",  "return_rate", "Deep discounts → more returns"),
    ("avg_price",     "order_count", "Price shock → order volume drop"),
    ("avg_qty",       "revenue",     "Bulk orders → revenue spike"),
    ("evt_add_to_cart","order_count","Cart activity → conversion"),
]


# ── Main causal engine ────────────────────────────────────────────────────────

class CausalInferenceEngine:

    def __init__(self, engine: Engine):
        self.engine = engine

    def analyze(self, anomaly: dict) -> dict:
        """
        Given an anomaly dict (from anomaly_events table), return causal findings.
        Returns: {cause, effect, ate, confidence, explanation}
        """
        region   = anomaly.get("region", "North")
        category = anomaly.get("category", "Electronics")
        metric   = anomaly.get("metric_name", "total_revenue")

        logger.info("Causal analysis for anomaly in %s/%s – metric=%s", region, category, metric)

        # Load feature data
        order_feat = _load_order_features(self.engine, region, category)
        event_feat = _load_event_features(self.engine)

        if order_feat.empty:
            return self._empty_result(anomaly, "insufficient_data")

        # Merge event features
        if not event_feat.empty:
            df = order_feat.merge(event_feat, on="hour_bucket", how="left").fillna(0)
        else:
            df = order_feat.copy()

        # Choose best candidate pairs for this metric
        best: Optional[dict] = None
        best_score = 0.0

        for treatment, outcome, hypothesis in CAUSAL_PAIRS:
            if treatment not in df.columns or outcome not in df.columns:
                continue

            if DOWHY_AVAILABLE:
                result = _dowhy_estimate(df, treatment, outcome)
            else:
                result = _regression_fallback(df, treatment, outcome)

            ate        = result["ate"]
            confidence = max(0.0, 1.0 - result["p_value"])

            if confidence > best_score:
                best_score = confidence
                best = {
                    "cause_variable":  treatment,
                    "effect_variable": outcome,
                    "ate":             ate,
                    "confidence":      confidence,
                    "method":          result["method"],
                    "hypothesis":      hypothesis,
                    "explanation": (
                        f"{treatment} caused {outcome} anomaly in {region}/{category} "
                        f"with ATE={ate:.4f} and confidence={confidence:.2%}. "
                        f"Hypothesis: {hypothesis}."
                    ),
                    "supporting_data": {
                        "region": region, "category": category,
                        "metric": metric,
                        "n_samples": len(df),
                        "refutation": result.get("refutation", ""),
                    },
                }

        if best is None:
            return self._empty_result(anomaly, "no_causal_pairs_matched")

        return best

    def persist(self, anomaly_id: int, finding: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO causal_findings
                    (anomaly_id, cause_variable, effect_variable,
                     ate, confidence, method, explanation, supporting_data)
                VALUES
                    (:anomaly_id, :cause_variable, :effect_variable,
                     :ate, :confidence, :method, :explanation,
                     CAST(:supporting_data AS jsonb))
            """), {
                "anomaly_id":     anomaly_id,
                "cause_variable": finding["cause_variable"],
                "effect_variable":finding["effect_variable"],
                "ate":            finding["ate"],
                "confidence":     finding["confidence"],
                "method":         finding["method"],
                "explanation":    finding["explanation"],
                "supporting_data":json.dumps(finding["supporting_data"]),
            })

    def _empty_result(self, anomaly: dict, reason: str) -> dict:
        return {
            "cause_variable":  "unknown",
            "effect_variable": anomaly.get("metric_name", "unknown"),
            "ate":             0.0,
            "confidence":      0.0,
            "method":          reason,
            "hypothesis":      "No causal structure identified",
            "explanation": (
                f"Causal analysis inconclusive for {anomaly.get('region')}"
                f"/{anomaly.get('category')}: {reason}"
            ),
            "supporting_data": {"anomaly": anomaly},
        }
