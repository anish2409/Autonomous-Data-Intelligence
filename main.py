"""
main.py
Autonomous Data Intelligence System – main entry point.

Modes:
  --mode seed       : Seed 30 days of historical data
  --mode run        : Single end-to-end intelligence cycle
  --mode daemon     : Continuous loop (production mode)
  --mode heal       : Run self-healing scan only
  --mode demo       : Demo with sample output (no DB required)
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("adi.main")

from config.settings import config


# ── Demo mode (no database needed) ───────────────────────────────────────────

def run_demo() -> None:
    """Print a rich sample output demonstrating the full system pipeline."""
    demo_output = {
        "pipeline": {
            "run_id": 42,
            "rows_ingested": 1000,
            "status": "SUCCESS",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "anomalies_detected": [
            {
                "anomaly_id": 1,
                "metric_name": "total_revenue",
                "metric_value": 98450.75,
                "expected_value": 12200.00,
                "z_score": 8.94,
                "severity": "CRITICAL",
                "region": "North",
                "category": "Electronics"
            },
            {
                "anomaly_id": 2,
                "metric_name": "return_rate",
                "metric_value": 0.43,
                "expected_value": 0.07,
                "z_score": 5.12,
                "severity": "HIGH",
                "region": "West",
                "category": "Apparel"
            }
        ],
        "causal_findings": [
            {
                "anomaly_id": 1,
                "cause_variable": "avg_discount",
                "effect_variable": "total_revenue",
                "ate": 124.57,
                "confidence": 0.87,
                "method": "dowhy_backdoor_lr",
                "explanation": "avg_discount caused total_revenue anomaly in North/Electronics with ATE=124.5700 and confidence=87.00%. Hypothesis: High discount → revenue anomaly."
            }
        ],
        "agent_decisions": [
            {
                "anomaly_id": 1,
                "priority": "P1",
                "final_action": "Freeze further discount promotions in North/Electronics for 4 hours.",
                "analyst_output": "ANALYST FINDING: The observed KPI deviation indicates a statistically significant anomaly (z-score=8.94, CRITICAL). Revenue spike of 8x normal in Electronics/North, consistent with a flash-sale event or bot-driven bulk order activity.",
                "causal_output": "CAUSAL FINDING: avg_discount (treatment) causally drives total_revenue (outcome) with ATE=124.57 and confidence=87%. The discount lever triggered a non-linear demand surge, likely amplified by inventory scarcity signals.",
                "decision_output": "DECISION [P1]: 1) Freeze discount promotions in North/Electronics for 4h. 2) Alert inventory team – stockout risk within 2h. 3) Activate checkout API rate-limiting (max 10 req/s per customer). 4) Notify fraud team for bot-order review. 5) Schedule post-mortem T+24h. Monitor: cart abandonment rate, API error rate, inventory levels."
            }
        ],
        "self_healing": {
            "drifts_detected": 0,
            "drifts_healed": 0,
            "pipeline_health": "HEALTHY"
        }
    }

    separator = "═" * 70
    print(f"\n{separator}")
    print("  🤖  AUTONOMOUS DATA INTELLIGENCE SYSTEM  — DEMO OUTPUT")
    print(separator)

    print("\n📥  INGESTION PIPELINE")
    p = demo_output["pipeline"]
    print(f"   Run ID:        {p['run_id']}")
    print(f"   Rows ingested: {p['rows_ingested']:,}")
    print(f"   Status:        {p['status']}")
    print(f"   Timestamp:     {p['timestamp']}")

    print(f"\n🔍  ANOMALY DETECTION ({len(demo_output['anomalies_detected'])} found)")
    for a in demo_output["anomalies_detected"]:
        print(f"\n   [{a['severity']}] {a['metric_name']}")
        print(f"   Region/Category: {a['region']} / {a['category']}")
        print(f"   Value:     {a['metric_value']:,.2f}  (expected {a['expected_value']:,.2f})")
        print(f"   Z-score:   {a['z_score']:.2f}σ")

    print(f"\n🔗  CAUSAL INFERENCE")
    for f in demo_output["causal_findings"]:
        print(f"\n   {f['cause_variable']} → {f['effect_variable']}")
        print(f"   ATE:        {f['ate']:+.4f}")
        print(f"   Confidence: {f['confidence']:.0%}")
        print(f"   Method:     {f['method']}")
        print(f"   Explanation: {f['explanation']}")

    print(f"\n🤝  MULTI-AGENT DECISIONS")
    for d in demo_output["agent_decisions"]:
        print(f"\n   Priority:  {d['priority']}")
        print(f"   Action:    {d['final_action']}")
        print(f"\n   [Analyst] {d['analyst_output'][:180]}…")
        print(f"\n   [Causal]  {d['causal_output'][:180]}…")
        print(f"\n   [Decision]{d['decision_output'][:300]}…")

    print(f"\n🔧  SELF-HEALING")
    sh = demo_output["self_healing"]
    print(f"   Drifts detected: {sh['drifts_detected']}")
    print(f"   Drifts healed:   {sh['drifts_healed']}")
    print(f"   Pipeline health: {sh['pipeline_health']}")

    print(f"\n{separator}\n")

    # Write sample output to file
    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, "sample_output.json")
    with open(out_path, "w") as f:
        json.dump(demo_output, f, indent=2, default=str)
    print(f"  ✅  Sample output written to: {out_path}\n")


# ── Full pipeline cycle ────────────────────────────────────────────────────────

def run_cycle(engine) -> dict:
    from ingestion.pipeline import run_ingestion_cycle
    from anomaly.detector import AnomalyDetectionEngine
    from causal.inference import CausalInferenceEngine
    from agents.orchestrator import AgentOrchestrator
    from self_healing.healer import SelfHealingPipeline

    results = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # 1. Ingest
    logger.info("── STEP 1: Data Ingestion ──")
    ingest_result = run_ingestion_cycle(engine, batch_n=config.pipeline.batch_size)
    results["ingestion"] = ingest_result

    # 2. Detect anomalies
    logger.info("── STEP 2: Anomaly Detection ──")
    detector   = AnomalyDetectionEngine(engine)
    anomalies  = detector.run()
    results["anomalies"] = anomalies
    logger.info("  Found %d anomalies", len(anomalies))

    # 3. Causal + Agent analysis (top anomalies only)
    causal_engine = CausalInferenceEngine(engine)
    orchestrator  = AgentOrchestrator(engine)
    decisions     = []

    for anomaly in anomalies[:3]:   # cap at top-3 to manage API cost
        logger.info("── STEP 3+4: Causal + Agent for anomaly %s ──",
                    anomaly.get("metric_name"))
        finding  = causal_engine.analyze(anomaly)
        if anomaly.get("anomaly_id"):
            causal_engine.persist(anomaly["anomaly_id"], finding)

        decision = orchestrator.run(anomaly, finding)
        decisions.append({
            "anomaly_id":     decision.anomaly_id,
            "priority":       decision.priority,
            "final_action":   decision.final_action,
        })

    results["decisions"] = decisions

    # 5. Self-heal
    logger.info("── STEP 5: Self-Healing ──")
    healer       = SelfHealingPipeline(engine)
    heal_results = healer.run()
    results["healing"] = {
        "drifts": len(heal_results),
        "healed": sum(1 for r in heal_results if r.healed)
    }

    return results


def run_daemon(engine) -> None:
    interval = config.pipeline.ingestion_interval_sec
    logger.info("Starting daemon loop (interval=%ds)", interval)
    while True:
        try:
            results = run_cycle(engine)
            logger.info("Cycle complete: %s", json.dumps(results, default=str))
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            break
        except Exception as exc:
            logger.exception("Cycle error: %s", exc)
        time.sleep(interval)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Data Intelligence System")
    parser.add_argument("--mode", choices=["seed", "run", "daemon", "heal", "demo"],
                        default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
        return

    # All other modes require a DB connection
    from sqlalchemy import create_engine as sa_create_engine
    engine = sa_create_engine(
        config.db.url,
        pool_size=config.db.pool_size,
        max_overflow=config.db.max_overflow,
        pool_pre_ping=True,
    )

    if args.mode == "seed":
        from ingestion.pipeline import seed_historical
        seed_historical(engine, days=30)

    elif args.mode == "run":
        results = run_cycle(engine)
        print(json.dumps(results, indent=2, default=str))

    elif args.mode == "daemon":
        run_daemon(engine)

    elif args.mode == "heal":
        from self_healing.healer import SelfHealingPipeline
        healer  = SelfHealingPipeline(engine)
        results = healer.run()
        report  = healer.health_report()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
