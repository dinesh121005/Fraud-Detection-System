"""
FinPulse Persistence Layer Configuration.

Supports environment-driven configuration for PostgreSQL (primary runtime)
and SQLite (isolated test fallback).
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL connection & pool configuration."""
    host: str = "localhost"
    port: int = 5432
    dbname: str = "finpulse"
    user: str = "finpulse_admin"
    password: str = "finpulse_secret_2026"
    min_connections: int = 2
    max_connections: int = 20
    connect_timeout: int = 5
    ssl_mode: str = "prefer"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Instantiate PostgresConfig from environment variables."""
        db_url = os.environ.get("FINPULSE_DB_URL") or os.environ.get("DATABASE_URL")
        if db_url and db_url.startswith(("postgres://", "postgresql://")):
            # Parse connection URL
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                dbname=parsed.path.lstrip("/") or "finpulse",
                user=parsed.username or "finpulse_admin",
                password=parsed.password or "finpulse_secret_2026",
                min_connections=int(os.environ.get("POSTGRES_MIN_CONNECTIONS", "2")),
                max_connections=int(os.environ.get("POSTGRES_MAX_CONNECTIONS", "20")),
                connect_timeout=int(os.environ.get("POSTGRES_CONNECT_TIMEOUT", "5")),
                ssl_mode=os.environ.get("POSTGRES_SSL_MODE", "prefer"),
            )

        return cls(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "finpulse"),
            user=os.environ.get("POSTGRES_USER", "finpulse_admin"),
            password=os.environ.get("POSTGRES_PASSWORD", "finpulse_secret_2026"),
            min_connections=int(os.environ.get("POSTGRES_MIN_CONNECTIONS", "2")),
            max_connections=int(os.environ.get("POSTGRES_MAX_CONNECTIONS", "20")),
            connect_timeout=int(os.environ.get("POSTGRES_CONNECT_TIMEOUT", "5")),
            ssl_mode=os.environ.get("POSTGRES_SSL_MODE", "prefer"),
        )

    @property
    def connection_kwargs(self) -> dict:
        """Return kwargs dict suitable for psycopg2.connect."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "connect_timeout": self.connect_timeout,
        }
        if self.ssl_mode and self.ssl_mode != "disable":
            kwargs["sslmode"] = self.ssl_mode
        return kwargs

    @property
    def dsn(self) -> str:
        """Return sanitized DSN string (with password masked for logging)."""
        return f"postgresql://{self.user}:****@{self.host}:{self.port}/{self.dbname}"


def get_persistence_backend(explicit_db_path: Optional[str] = None) -> str:
    """
    Determine the persistence backend: 'postgres' or 'sqlite'.
    Rules:
    1. If explicit_db_path is ':memory:' or ends with '.db', return 'sqlite'.
    2. If explicit_db_path starts with 'postgresql://' or 'postgres://', return 'postgres'.
    3. If FINPULSE_PERSISTENCE_BACKEND env var is set, return its value ('postgres' or 'sqlite').
    4. Default to 'postgres' for production/runtime if postgres is reachable; otherwise gracefully fallback to 'sqlite'.
    """
    if explicit_db_path:
        if explicit_db_path == ":memory:" or explicit_db_path.endswith(".db"):
            return "sqlite"
        if explicit_db_path.startswith(("postgres://", "postgresql://")):
            return "postgres"

    env_backend = os.environ.get("FINPULSE_PERSISTENCE_BACKEND")
    if env_backend:
        return env_backend.lower()

    return "postgres"
