"""
FinPulse PostgreSQL Connection & Pooling Layer.

Provides thread-safe connection pooling, health checks, automatic rollback on error,
and resilient retry/reconnect semantics for FinPulse.
"""

import time
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator

try:
    import psycopg2
    from psycopg2 import pool, extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from src.persistence.config import PostgresConfig
from src.monitoring.logger import get_logger
from src.monitoring.metrics import record_error

logger = get_logger("FinPulse.Persistence.Connection")


class PostgresConnectionPool:
    """
    Thread-safe PostgreSQL connection pool wrapper using psycopg2.pool.ThreadedConnectionPool.
    Includes health probes, automatic connection checkout/checkin, and error handling.
    """

    _instance: Optional["PostgresConnectionPool"] = None
    _lock = threading.RLock()

    def __init__(self, config: Optional[PostgresConfig] = None):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2-binary is not installed. Install it with `pip install psycopg2-binary`.")

        self.config = config or PostgresConfig.from_env()
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._is_closed = False
        self._initialize_pool()

    def _initialize_pool(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """Initialize the connection pool with retry on transient startup failures."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self._pool = pool.ThreadedConnectionPool(
                    minconn=self.config.min_connections,
                    maxconn=self.config.max_connections,
                    **self.config.connection_kwargs
                )
                self._is_closed = False
                logger.info(
                    "PostgreSQL connection pool initialized successfully",
                    dsn=self.config.dsn,
                    min_conn=self.config.min_connections,
                    max_conn=self.config.max_connections,
                )
                return
            except Exception as e:
                last_error = e
                record_error("persistence", "pool_init_retry")
                logger.warning(
                    f"PostgreSQL connection attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2

        record_error("persistence", "pool_init_failed")
        logger.error("Failed to initialize PostgreSQL connection pool", error=str(last_error))
        raise ConnectionError(f"Could not connect to PostgreSQL at {self.config.dsn}: {last_error}")

    @classmethod
    def get_instance(cls, config: Optional[PostgresConfig] = None) -> "PostgresConnectionPool":
        """Singleton accessor for application-wide shared connection pool."""
        with cls._lock:
            if cls._instance is None or cls._instance._is_closed:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful in tests and restart scenarios)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Context manager to borrow a connection from the pool.
        Validates connection liveness, commits on normal exit, rolls back on error,
        and automatically handles server restarts.
        """
        if self._is_closed or self._pool is None:
            self._initialize_pool()

        conn = None
        # Attempt to borrow a live connection
        for attempt in range(2):
            try:
                conn = self._pool.getconn()
                if conn is None:
                    raise ConnectionError("PostgreSQL pool exhausted: no available connections.")
                # Ensure in valid non-autocommit state and verify liveness
                conn.autocommit = False
                with conn.cursor() as probe_cur:
                    probe_cur.execute("SELECT 1;")
                conn.rollback()
                break
            except Exception:
                if conn:
                    try:
                        self._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = None
                self._initialize_pool()
                if attempt == 1:
                    raise

        try:
            yield conn
            conn.commit()
        except Exception as ex:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            record_error("persistence", "transaction_error")
            logger.error("Database operation failed, rolled back", error=str(ex))
            raise
        finally:
            if conn is not None and self._pool is not None:
                try:
                    self._pool.putconn(conn)
                except Exception as put_err:
                    logger.warning("Error returning connection to pool", error=str(put_err))

    def check_health(self) -> Dict[str, Any]:
        """
        Verify database connectivity and return health telemetry.
        Executes a lightweight 'SELECT 1' and queries database version.
        """
        t0 = time.perf_counter()
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
                    cur.execute("SELECT version();")
                    ver = cur.fetchone()[0]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "HEALTHY",
                "backend": "postgresql",
                "latency_ms": round(latency_ms, 2),
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.dbname,
                "user": self.config.user,
                "version": ver.split()[0] + " " + ver.split()[1] if ver else "PostgreSQL",
                "pool_min": self.config.min_connections,
                "pool_max": self.config.max_connections,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            record_error("persistence", "health_check_failed")
            return {
                "status": "UNHEALTHY",
                "backend": "postgresql",
                "latency_ms": round(latency_ms, 2),
                "error": str(e),
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.dbname,
            }

    def close(self) -> None:
        """Close all connections in the pool cleanly."""
        with self._lock:
            if not self._is_closed and self._pool is not None:
                try:
                    self._pool.closeall()
                    logger.info("PostgreSQL connection pool closed successfully.")
                except Exception as e:
                    logger.warning("Error closing connection pool", error=str(e))
                finally:
                    self._pool = None
                    self._is_closed = True
