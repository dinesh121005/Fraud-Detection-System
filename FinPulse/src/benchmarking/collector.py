"""FinPulse R6.2 — Resource & Telemetry Collector.

Captures:
- Docker container CPU & memory stats (Kafka, Redis)
- Worker process CPU & memory (psutil)
- Host-level CPU & memory
- Latency percentiles (P50, P95, P99, Mean) across stages
- Kafka consumer lag progression over time
- R6.1 Prometheus metrics snapshots
"""

import os
import time
import json
import subprocess
import numpy as np
import psutil
from typing import Dict, Any, List, Optional


class ResourceCollector:
    """Collects container, process, and system hardware resource metrics."""

    def __init__(self, target_containers: Optional[List[str]] = None):
        self.target_containers = target_containers or ["finpulse-kafka", "finpulse-redis"]
        self.process = psutil.Process(os.getpid())

    def get_container_stats(self) -> Dict[str, Dict[str, Any]]:
        """Collect real-time Docker container CPU and memory stats via docker stats."""
        stats = {}
        try:
            cmd = ["docker", "stats", "--no-stream", "--format", "json"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        name = data.get("Name", "")
                        for target in self.target_containers:
                            if target in name:
                                stats[target] = {
                                    "cpu_perc": data.get("CPUPerc", "0.0%"),
                                    "mem_usage": data.get("MemUsage", "0B / 0B"),
                                    "mem_perc": data.get("MemPerc", "0.0%"),
                                    "pids": data.get("PIDs", "0"),
                                    "net_io": data.get("NetIO", "0B / 0B"),
                                }
                    except Exception:
                        pass
        except Exception:
            pass
        return stats

    def get_process_stats(self) -> Dict[str, Any]:
        """Collect host worker process resource utilization."""
        try:
            mem_info = self.process.memory_info()
            return {
                "cpu_percent": self.process.cpu_percent(interval=None),
                "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                "num_threads": self.process.num_threads(),
            }
        except Exception:
            return {"cpu_percent": 0.0, "rss_mb": 0.0, "num_threads": 1}

    def get_host_stats(self) -> Dict[str, Any]:
        """Collect overall system CPU and memory metrics."""
        vm = psutil.virtual_memory()
        return {
            "host_cpu_percent": psutil.cpu_percent(interval=None),
            "host_mem_used_gb": round((vm.total - vm.available) / (1024 ** 3), 2),
            "host_mem_total_gb": round(vm.total / (1024 ** 3), 2),
            "host_mem_percent": vm.percent,
        }

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture complete resource snapshot across containers and host."""
        return {
            "containers": self.get_container_stats(),
            "worker_process": self.get_process_stats(),
            "host": self.get_host_stats(),
            "timestamp": time.time(),
        }


class TelemetryCollector:
    """Aggregates latency observations, Kafka lag samples, and Prometheus counters."""

    def __init__(self):
        self.stage_latencies: Dict[str, List[float]] = {
            "e2e": [],
            "redis": [],
            "feature_engine": [],
            "model": [],
            "risk_engine": [],
            "serialization": [],
            "kafka_publish": [],
        }
        self.lag_samples: List[int] = []

    def record_stage_latency(self, stage: str, latency_ms: float):
        """Append latency observation for a specific stage."""
        if stage in self.stage_latencies:
            self.stage_latencies[stage].append(float(latency_ms))

    def record_lag_sample(self, lag: int):
        """Append a Kafka consumer lag sample."""
        self.lag_samples.append(max(0, int(lag)))

    def calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate standard latency distribution statistics."""
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "count": int(len(arr)),
        }

    def summarize_latencies(self) -> Dict[str, Dict[str, float]]:
        """Return distribution summaries for all tracked stages."""
        return {
            stage: self.calculate_distribution(lat_list)
            for stage, lat_list in self.stage_latencies.items()
        }

    def summarize_lag(self) -> Dict[str, Any]:
        """Summarize Kafka consumer lag metrics."""
        if not self.lag_samples:
            return {"initial_lag": 0, "max_lag": 0, "final_lag": 0, "trend": "stable", "samples_count": 0}
        initial = self.lag_samples[0]
        final = self.lag_samples[-1]
        max_lag = max(self.lag_samples)
        trend = "stable"
        if final > initial and final > 10:
            trend = "growing"
        elif final < max_lag and final <= 2:
            trend = "drained_to_baseline"

        return {
            "initial_lag": initial,
            "max_lag": max_lag,
            "final_lag": final,
            "trend": trend,
            "samples_count": len(self.lag_samples),
        }
