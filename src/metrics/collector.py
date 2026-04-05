"""Metrics collection and reporting for fever-prediction-platform."""
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class FeverPredictionPlatformMetrics:
    """Collect, aggregate, and report metrics for fever-prediction-platform operations."""

    def __init__(self, namespace: str = "fever-prediction-platform"):
        self.namespace = namespace
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, float] = {}
        self._start_time = time.time()

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        self._metrics[name].append(value)

    def increment(self, name: str, amount: int = 1):
        self._counters[name] += amount

    def start_timer(self, name: str):
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        if name not in self._timers:
            return 0.0
        elapsed = time.time() - self._timers.pop(name)
        self.record(f"{name}_duration_seconds", elapsed)
        return elapsed

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "namespace": self.namespace,
            "uptime_seconds": time.time() - self._start_time,
            "counters": dict(self._counters),
            "gauges": {},
        }
        for name, values in self._metrics.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                summary["gauges"][name] = {
                    "count": n,
                    "mean": sum(sorted_v) / n,
                    "min": sorted_v[0],
                    "max": sorted_v[-1],
                    "p50": sorted_v[n // 2],
                    "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
                    "p99": sorted_v[int(n * 0.99)] if n > 1 else sorted_v[0],
                }
        return summary

    def reset(self):
        self._metrics.clear()
        self._counters.clear()
        self._timers.clear()
        self._start_time = time.time()
