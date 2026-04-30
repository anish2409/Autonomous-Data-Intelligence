"""
agents/orchestrator.py
Multi-agent reasoning system powered by the Anthropic API.

Agents:
  1. Analyst Agent   – interprets the anomaly signal
  2. Causal Agent    – explains the root cause
  3. Decision Agent  – recommends concrete actions

They engage in a structured debate (configurable rounds) before the
Decision Agent synthesises a final recommendation.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import config

logger = logging.getLogger("adi.agents")

# ── API client (thin wrapper – no LangChain dep required) ────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def _call_llm(system_prompt: str, messages: list[dict], max_tokens: int = None) -> str:
    """Call Anthropic Messages API. Falls back to deterministic mock if key absent."""
    if not ANTHROPIC_API_KEY:
        return _mock_llm_response(system_prompt, messages)

    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      config.agents.model,
        "max_tokens": max_tokens or config.agents.max_tokens,
        "system":     system_prompt,
        "messages":   messages,
    }
    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers,
                             json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()
    except Exception as exc:
        logger.warning("LLM API call failed (%s); using mock response", exc)
        return _mock_llm_response(system_prompt, messages)


def _mock_llm_response(system_prompt: str, messages: list[dict]) -> str:
    """Deterministic stub for CI / offline environments."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if "ANALYST" in system_prompt or "anomaly" in last_user.lower():
        return (
            "ANALYST FINDING: The observed KPI deviation indicates a statistically "
            "significant anomaly (z-score > 3σ). The spike in revenue is localised to "
            "the Electronics category in the North region, occurring within a 2-hour "
            "window. Pattern is consistent with a flash-sale event or bot-driven bulk order."
        )
    if "CAUSAL" in system_prompt or "cause" in last_user.lower():
        return (
            "CAUSAL FINDING: Root-cause analysis confirms that avg_discount (treatment) "
            "has a strong causal effect on revenue (outcome) with ATE=124.5 and "
            "confidence=87%. The discount lever triggered a non-linear demand surge, "
            "amplified by low inventory constraints. Confidence: HIGH."
        )
    return (
        "DECISION: Recommend immediate P1 action. "
        "1) Freeze further discount promotions in North/Electronics for 4 hours. "
        "2) Alert inventory team to prevent stockout. "
        "3) Activate rate-limiting on checkout API to prevent bot orders. "
        "4) Schedule post-mortem within 24 hours. ETA to resolve: 2 hours."
    )


# ── Agent definitions ─────────────────────────────────────────────────────────

ANALYST_SYSTEM = """You are the Analyst Agent in a Data Intelligence System.
Your role: interpret anomaly signals from KPI monitoring dashboards.
Given raw anomaly data, you must:
1. Characterise the anomaly (type, magnitude, affected dimensions)
2. Assess whether it is a true anomaly or statistical noise
3. Identify which business process is most likely affected
4. Output a structured ANALYST FINDING paragraph.
Be concise, precise, and data-driven. Avoid speculation without evidence."""

CAUSAL_SYSTEM = """You are the Causal Agent in a Data Intelligence System.
Your role: explain WHY an anomaly occurred using causal analysis results.
Given anomaly characterisation and causal inference output, you must:
1. State the primary cause-effect relationship
2. Quantify the causal effect (ATE) and confidence
3. Rule out confounders
4. Output a structured CAUSAL FINDING paragraph.
Use language like "X caused Y with confidence Z% and ATE=N"."""

DECISION_SYSTEM = """You are the Decision Agent in a Data Intelligence System.
Your role: synthesise analyst and causal findings into actionable recommendations.
Given both prior agent outputs, you must:
1. Assign a priority: P0 (critical), P1 (urgent), P2 (moderate), P3 (low)
2. Recommend 3-5 specific, measurable actions with owners and ETAs
3. Identify leading indicators to monitor post-action
4. Output a structured DECISION paragraph.
Think like a VP of Engineering + VP of Data combined."""


# ── Agent debate orchestrator ─────────────────────────────────────────────────

@dataclass
class AgentDebateResult:
    anomaly_id:      int
    analyst_output:  str
    causal_output:   str
    decision_output: str
    final_action:    str
    priority:        str
    debate_history:  list[dict] = field(default_factory=list)


def _extract_priority(text: str) -> str:
    for p in ["P0", "P1", "P2", "P3"]:
        if p in text:
            return p
    return "P2"


def _extract_action(text: str) -> str:
    """Pull the first recommended action from decision text."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("1)") or line.startswith("1."):
            return line[2:].strip()[:128]
    return text[:128].strip()


class AgentOrchestrator:

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, anomaly: dict, causal_finding: dict) -> AgentDebateResult:
        """
        Orchestrate a multi-round debate between the three agents.
        anomaly:        dict from anomaly_events
        causal_finding: dict from CausalInferenceEngine.analyze()
        """
        anomaly_id  = anomaly.get("anomaly_id", 0)
        history: list[dict] = []

        # ── Round 0: prime context ──────────────────────────────────────────
        context_msg = (
            f"ANOMALY REPORT:\n"
            f"  Metric:    {anomaly.get('metric_name')}\n"
            f"  Value:     {anomaly.get('metric_value')}\n"
            f"  Expected:  {anomaly.get('expected_value')}\n"
            f"  Z-score:   {anomaly.get('z_score')}\n"
            f"  Severity:  {anomaly.get('severity')}\n"
            f"  Region:    {anomaly.get('region')}\n"
            f"  Category:  {anomaly.get('category')}\n"
            f"  Detected:  {anomaly.get('detected_at')}\n\n"
            f"CAUSAL ANALYSIS:\n"
            f"  {causal_finding.get('explanation', 'No causal analysis available.')}"
        )

        # ── Analyst Agent ───────────────────────────────────────────────────
        logger.info("[Agent] Analyst Agent reasoning…")
        analyst_msgs = [{"role": "user", "content": context_msg}]
        analyst_out  = _call_llm(ANALYST_SYSTEM, analyst_msgs)
        history.append({"agent": "Analyst", "round": 1, "content": analyst_out})

        # ── Causal Agent ────────────────────────────────────────────────────
        logger.info("[Agent] Causal Agent reasoning…")
        causal_msgs = [
            {"role": "user", "content": context_msg},
            {"role": "assistant", "content": analyst_out},
            {"role": "user", "content":
                "Now provide your CAUSAL FINDING, building on the analyst's assessment. "
                "Do you agree with the root-cause hypothesis? Refine or challenge it."},
        ]
        causal_out = _call_llm(CAUSAL_SYSTEM, causal_msgs)
        history.append({"agent": "Causal", "round": 1, "content": causal_out})

        # ── Optional debate rounds ──────────────────────────────────────────
        for rnd in range(2, config.agents.debate_rounds + 1):
            logger.info("[Agent] Debate round %d…", rnd)
            rebuttal_msgs = [
                {"role": "user",      "content": context_msg},
                {"role": "assistant", "content": analyst_out},
                {"role": "user",      "content": causal_out},
                {"role": "assistant", "content":
                    f"Round {rnd}: Do you want to revise your finding in light of the causal evidence?"},
            ]
            revised = _call_llm(ANALYST_SYSTEM, rebuttal_msgs)
            if revised and len(revised) > 20:
                analyst_out = revised
                history.append({"agent": "Analyst", "round": rnd, "content": revised})

        # ── Decision Agent ──────────────────────────────────────────────────
        logger.info("[Agent] Decision Agent synthesising…")
        decision_msgs = [
            {"role": "user", "content": (
                f"{context_msg}\n\n"
                f"ANALYST FINDING:\n{analyst_out}\n\n"
                f"CAUSAL FINDING:\n{causal_out}\n\n"
                "Based on both findings, provide your DECISION with priority and actions."
            )},
        ]
        decision_out = _call_llm(DECISION_SYSTEM, decision_msgs)
        history.append({"agent": "Decision", "round": 1, "content": decision_out})

        priority     = _extract_priority(decision_out)
        final_action = _extract_action(decision_out)

        result = AgentDebateResult(
            anomaly_id=anomaly_id,
            analyst_output=analyst_out,
            causal_output=causal_out,
            decision_output=decision_out,
            final_action=final_action,
            priority=priority,
            debate_history=history,
        )

        self._persist(result)
        return result

    def _persist(self, result: AgentDebateResult) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO agent_decisions
                    (anomaly_id, analyst_output, causal_output,
                     decision_output, final_action, priority, metadata)
                VALUES
                    (:anomaly_id, :analyst_output, :causal_output,
                     :decision_output, :final_action, :priority,
                     CAST(:metadata AS jsonb))
            """), {
                "anomaly_id":     result.anomaly_id,
                "analyst_output": result.analyst_output,
                "causal_output":  result.causal_output,
                "decision_output":result.decision_output,
                "final_action":   result.final_action,
                "priority":       result.priority,
                "metadata":       json.dumps({"debate_history": result.debate_history}),
            })
        logger.info("[Agent] Decision persisted for anomaly_id=%d (priority=%s)",
                    result.anomaly_id, result.priority)
