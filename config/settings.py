"""
config/settings.py
Central configuration for the Autonomous Data Intelligence System.
All secrets are read from environment variables; defaults are for local dev.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    name: str = os.getenv("PGDATABASE", "adi_system")
    user: str = os.getenv("PGUSER","postgres" )
     #"adi_user" )
    password: str = os.getenv("PGPASSWORD", 241974)
                              #"adi_password")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))

    """@property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    """
    @property
    def url(self) -> str:
        return "postgresql+psycopg2://postgres:241974@localhost:5432/adi_system"
    
@dataclass
class AnomalyConfig:
    zscore_threshold: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
    isolation_contamination: float = float(os.getenv("ISOLATION_CONTAMINATION", "0.05"))
    rolling_window_hours: int = int(os.getenv("ROLLING_WINDOW_HOURS", "24"))
    min_samples_for_ml: int = int(os.getenv("MIN_SAMPLES_FOR_ML", "50"))
    severity_thresholds: dict = field(default_factory=lambda: {
        "LOW":      (3.0, 4.0),
        "MEDIUM":   (4.0, 6.0),
        "HIGH":     (6.0, 9.0),
        "CRITICAL": (9.0, float("inf")),
    })


@dataclass
class AgentConfig:
    model: str = os.getenv("OPENAI_MODEL", "claude-sonnet-4-20250514")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
    debate_rounds: int = int(os.getenv("AGENT_DEBATE_ROUNDS", "2"))


@dataclass
class PipelineConfig:
    batch_size: int = int(os.getenv("BATCH_SIZE", "1000"))
    ingestion_interval_sec: int = int(os.getenv("INGESTION_INTERVAL_SEC", "60"))
    schema_check_interval_sec: int = int(os.getenv("SCHEMA_CHECK_INTERVAL_SEC", "300"))
    simulate_streaming: bool = os.getenv("SIMULATE_STREAMING", "true").lower() == "true"
    seed_row_count: int = int(os.getenv("SEED_ROW_COUNT", "50000"))


@dataclass
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")


# Singleton
config = AppConfig()
