"""50Hz 벤치마킹 및 성능 측정 모듈.

Latency, throughput, jitter를 측정하여 동시 운영 성능 검증.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FrameMetrics:
    """단일 프레임의 성능 지표."""

    timestamp: float                    # 절대 시간 (unix)
    frame_id: int                       # 프레임 번호
    setpoint_receive_time: Optional[float] = None  # Setpoint 수신 시간
    action_dispatch_time: Optional[float] = None   # Action 발행 시간
    state_send_time: Optional[float] = None        # State 발행 시간
    state_recv_time: Optional[float] = None        # State 수신 시간 (controller)

    @property
    def e2e_latency(self) -> Optional[float]:
        """End-to-end latency (setpoint receive → action dispatch, ms)."""
        if self.setpoint_receive_time and self.action_dispatch_time:
            return (self.action_dispatch_time - self.setpoint_receive_time) * 1000
        return None

    @property
    def state_latency(self) -> Optional[float]:
        """State transmission latency (send → receive, ms)."""
        if self.state_send_time and self.state_recv_time:
            return (self.state_recv_time - self.state_send_time) * 1000
        return None


class FrequencyMonitor:
    """50Hz 동작 빈도 모니터링."""

    def __init__(self, target_hz: float = 50.0, window_size: int = 100):
        self.target_hz = target_hz
        self.target_period = 1.0 / target_hz
        self.window_size = window_size

        self._timestamps: deque = deque(maxlen=window_size)
        self._last_tick = None
        self._frame_count = 0

    def tick(self) -> None:
        """한 프레임 진행."""
        now = time.perf_counter()
        self._timestamps.append(now)
        self._frame_count += 1
        self._last_tick = now

    @property
    def current_hz(self) -> float:
        """현재 빈도 (Hz)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def frame_count(self) -> int:
        """총 프레임 수."""
        return self._frame_count

    @property
    def period_ms(self) -> float:
        """프레임 주기 (ms)."""
        if len(self._timestamps) < 2:
            return 0.0
        intervals = []
        for i in range(1, len(self._timestamps)):
            dt = (self._timestamps[i] - self._timestamps[i-1]) * 1000
            intervals.append(dt)
        return sum(intervals) / len(intervals) if intervals else 0.0

    @property
    def jitter_ms(self) -> float:
        """프레임 주기 표준편차 (ms)."""
        if len(self._timestamps) < 3:
            return 0.0
        intervals = []
        for i in range(1, len(self._timestamps)):
            dt = (self._timestamps[i] - self._timestamps[i-1]) * 1000
            intervals.append(dt)

        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        return variance ** 0.5


class LatencyTracker:
    """Latency 측정 및 통계."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)

    def record(self, latency_ms: float) -> None:
        """Latency 기록."""
        self._latencies.append(latency_ms)

    @property
    def mean_ms(self) -> float:
        """평균 latency (ms)."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def min_ms(self) -> float:
        """최소 latency (ms)."""
        return min(self._latencies) if self._latencies else 0.0

    @property
    def max_ms(self) -> float:
        """최대 latency (ms)."""
        return max(self._latencies) if self._latencies else 0.0

    def percentile_ms(self, p: float = 95.0) -> float:
        """Percentile latency (ms)."""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * (p / 100.0))
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class BenchmarkSession:
    """전체 벤치마킹 세션 관리."""

    def __init__(self, output_dir: Optional[str] = None, session_name: str = "bench"):
        self.output_dir = Path(output_dir or "logs")
        self.session_name = session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frequency_monitor = FrequencyMonitor(target_hz=50.0)
        self.e2e_latency_tracker = LatencyTracker()
        self.state_latency_tracker = LatencyTracker()

        self._frames: list[FrameMetrics] = []
        self._start_time = time.time()

    def record_frame(self, metrics: FrameMetrics) -> None:
        """프레임 메트릭 기록."""
        self._frames.append(metrics)
        self.frequency_monitor.tick()

        if metrics.e2e_latency is not None:
            self.e2e_latency_tracker.record(metrics.e2e_latency)

        if metrics.state_latency is not None:
            self.state_latency_tracker.record(metrics.state_latency)

    def print_summary(self) -> str:
        """현재 통계 출력."""
        lines = [
            "=" * 60,
            f"Real2Sim Benchmark Summary ({self.session_name})",
            "=" * 60,
            f"Duration: {time.time() - self._start_time:.2f} sec",
            f"Total Frames: {self.frequency_monitor.frame_count}",
            "",
            "[Frequency (50Hz Target)]",
            f"  Current Hz: {self.frequency_monitor.current_hz:.2f}",
            f"  Period: {self.frequency_monitor.period_ms:.2f} ms",
            f"  Jitter (σ): {self.frequency_monitor.jitter_ms:.2f} ms",
            "",
            "[E2E Latency (Setpoint → Action)]",
            f"  Mean: {self.e2e_latency_tracker.mean_ms:.2f} ms",
            f"  Min: {self.e2e_latency_tracker.min_ms:.2f} ms",
            f"  Max: {self.e2e_latency_tracker.max_ms:.2f} ms",
            f"  p95: {self.e2e_latency_tracker.percentile_ms(95.0):.2f} ms",
            "",
            "[State Latency (Send → Receive)]",
            f"  Mean: {self.state_latency_tracker.mean_ms:.2f} ms",
            f"  Min: {self.state_latency_tracker.min_ms:.2f} ms",
            f"  Max: {self.state_latency_tracker.max_ms:.2f} ms",
            f"  p95: {self.state_latency_tracker.percentile_ms(95.0):.2f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)

    def save_results(self) -> Path:
        """결과를 JSON으로 저장."""
        result = {
            "session_name": self.session_name,
            "start_time": self._start_time,
            "duration_sec": time.time() - self._start_time,
            "total_frames": self.frequency_monitor.frame_count,
            "frequency": {
                "target_hz": 50.0,
                "current_hz": self.frequency_monitor.current_hz,
                "period_ms": self.frequency_monitor.period_ms,
                "jitter_ms": self.frequency_monitor.jitter_ms,
            },
            "e2e_latency_ms": {
                "mean": self.e2e_latency_tracker.mean_ms,
                "min": self.e2e_latency_tracker.min_ms,
                "max": self.e2e_latency_tracker.max_ms,
                "p95": self.e2e_latency_tracker.percentile_ms(95.0),
            },
            "state_latency_ms": {
                "mean": self.state_latency_tracker.mean_ms,
                "min": self.state_latency_tracker.min_ms,
                "max": self.state_latency_tracker.max_ms,
                "p95": self.state_latency_tracker.percentile_ms(95.0),
            },
        }

        output_path = self.output_dir / f"{self.session_name}_benchmark.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return output_path


if __name__ == "__main__":
    # 테스트: 간단한 50Hz 시뮬레이션
    bench = BenchmarkSession(session_name="test_50hz")

    start = time.perf_counter()
    for i in range(250):  # 5초 @ 50Hz
        frame = FrameMetrics(
            timestamp=time.time(),
            frame_id=i,
            setpoint_receive_time=time.perf_counter(),
            action_dispatch_time=time.perf_counter() + 0.001,
            state_send_time=time.perf_counter() + 0.002,
            state_recv_time=time.perf_counter() + 0.005,
        )
        bench.record_frame(frame)

        # 50Hz 유지
        elapsed = time.perf_counter() - start
        expected = i * 0.02
        if elapsed < expected:
            time.sleep(expected - elapsed)

    print(bench.print_summary())
    result_path = bench.save_results()
    print(f"\n결과 저장: {result_path}")
