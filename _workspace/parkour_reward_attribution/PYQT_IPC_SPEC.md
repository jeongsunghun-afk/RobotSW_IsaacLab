# PyQt + IPC Live Reward Attribution Viewer — Specification

**Status:** Design only. No implementation. References DESIGN.md §9 and expands into a complete specification for future stage-3+ development.

**Date:** 2026-05-27

**Scope:** Standalone live visualization of per-step reward breakdowns and foot contact during Go2 Parkour terrain trials, running in a separate process from the Isaac Sim environment.

---

## 1. Why matplotlib live visualization fails in IsaacLab

### Root causes

The user's prior attempt to render live matplotlib plots **during** a training or eval loop reported it "didn't work" (park-rl repository git history). Three entangled factors are responsible:

#### 1.1 Python GIL contention and blocking draws
- `plt.draw()`, `Figure.canvas.draw_idle()`, and `plt.pause()` in matplotlib are **blocking** calls on most backends. Even a 5–10 ms draw operation per 20 ms step (50 Hz policy) consumes 25–50% of the per-step budget.
- The Isaac Sim physics loop runs in a separate C++ thread, but the Python-side callbacks (e.g. `_get_observations()`, `_get_rewards()`) acquire the GIL. A long-running matplotlib draw will hold the GIL and starve the sim's Python update path, causing frame drops or policy latency spikes.

#### 1.2 Qt event loop collision with Omniverse Kit
- Omniverse Kit (the Isaac Sim engine) **already owns a PySide2/PySide6 Qt main event loop** (lines 421–432 in the original DESIGN.md show Kit's Qt loop is embedded).
- If the same Python process tries to spawn a second Qt main loop (e.g., via `matplotlib.use('QtAgg')` or explicit PyQt), a runtime conflict occurs: two event loops competing for control, windows freeze, redraws queue up, and the sim's renderpass stalls.
- Switching to a non-interactive backend (e.g., `Agg`, `Cairo`) sidesteps the Qt conflict but reintroduces the blocking-draw problem.

#### 1.3 Environmental dependencies and import conflicts
- The `isaac-parkour` conda environment ships a minimal PyTorch + Isaac Sim + rsl_rl stack. It **does not include PyQt/PySide** (to avoid bloating a 4 GB env).
- matplotlib's auto-backend detection (`matplotlib.use()` deferred or omitted) will fail gracefully if no GUI backend is available, but forcing `QtAgg` will trigger an ImportError or collide with Kit's existing PySide.
- Adding PyQt to the isaac-parkour env solely to support matplotlib in-process is a large footprint increase and introduces cross-process Qt initialization complexity.

### Why process isolation solves all three

**Process isolation — running the viewer in a separate Python process — breaks the coupling:**
1. The sim process owns no Qt event loop; matplotlib is not involved. The policy loop runs unimpeded.
2. The viewer process owns its own Python interpreter, GIL, and Qt main loop. No collision with Kit.
3. The sim process publishes data over an IPC channel (ZeroMQ or Unix socket). The viewer subscribes independently. No environment bloat.

As a result, the viewer can use a full-featured GUI toolkit (PyQt6) and a real-time plotting library (pyqtgraph) without affecting the sim's determinism or throughput.

---

## 2. Process topology

### Text diagram

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ Process A — IsaacLab sim (main training/eval entry point)       │
   │ Cmd: ./isaaclab.sh -p play_reward_attribution.py                │
   │ PID: (e.g. 12345)                                               │
   │                                                                 │
   │  ┌────────────────────────────────────────────────────────────┐│
   │  │ Environment loop (50 Hz policy step rate)                 ││
   │  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐││
   │  │  │ Go2Parkour   │→ │ _get_rewards()   │→ │ ZMQ PUB socket ││
   │  │  │ DirectRL     │  │ + contact read   │  │ (non-blocking) ││
   │  │  │ env          │  │ + msgpack encode │  │ ipc:///tmp/    ││
   │  │  │ (N=5, T=1k)  │  │                  │  │ parkour_reward││
   │  │  └──────────────┘  └──────────────────┘  └────────┬───────┘││
   │  │                                                   │         ││
   │  │ Payloads:                                         │         ││
   │  │ - rewards: [N=5, K=16] float32/step              │         ││
   │  │ - contact: [N=5, 4] bool/step                    │         ││
   │  │ - commands: [N=5, 3] float32/step (optional)     │         ││
   │  │ - metadata: step_index, done flags               │         ││
   │  │                                                   │         ││
   │  └───────────────────────────────────────────────────┼─────────┘│
   │                                                      │           │
   └──────────────────────────────────────────────────────┼───────────┘
                                                         │
                    ╔════════════════════════════════════╩══════════════════╗
                    ║ IPC Transport: ZeroMQ over Unix domain socket         ║
                    ║ Endpoint: ipc:///tmp/parkour_reward.sock (or tmpfs)   ║
                    ║ Mode: PUB/SUB (asynchronous, non-blocking)            ║
                    ║ High-water-mark: 10 messages (drops old if backlog)   ║
                    ║ Message rate: 50 Hz (sim step), payload ~500 B        ║
                    ╚════════════════════════════════════╦══════════════════╝
                                                         │
   ┌──────────────────────────────────────────────────────┼───────────┐
   │ Process B — reward_attribution_viewer.py              ↓          │
   │ Cmd: python reward_attribution_viewer.py              │          │
   │ PID: (separate, e.g. 12346)                          │          │
   │                                                       │          │
   │  ┌──────────────────────────────────────────────────┐│          │
   │  │ Receiver loop (Qt event loop, 30 Hz tick)        ││          │
   │  │  ┌─────────────────────────────────────────────┐││          │
   │  │  │ ZMQ SUB socket                              │││          │
   │  │  │ (non-blocking recv on Qt event loop)        │││          │
   │  │  │ ipc:///tmp/parkour_reward.sock              │││          │
   │  │  └────────────────┬────────────────────────────┘││          │
   │  │                   ↓                              ││          │
   │  │  ┌──────────────────────────────────────────────┐││          │
   │  │  │ QSocketNotifier                              │││          │
   │  │  │ (integrate ZMQ FD into Qt event loop)        │││          │
   │  │  │ Triggers on msg arrival                      │││          │
   │  │  └────────────────┬────────────────────────────┘││          │
   │  │                   ↓                              ││          │
   │  │  ┌──────────────────────────────────────────────┐││          │
   │  │  │ Ring buffers + state                         │││          │
   │  │  │ rewards:     [max_T=500, K=16]  float32      │││          │
   │  │  │ contact:     [max_T=500, 4]     bool         │││          │
   │  │  │ timestamps:  [max_T=500]        float64      │││          │
   │  │  │ env_selector: int (0–4)                      │││          │
   │  │  │ scroll_pos:   int (ring buffer tail)         │││          │
   │  │  └────────────────┬────────────────────────────┘││          │
   │  │                   ↓                              ││          │
   │  │  ┌──────────────────────────────────────────────┐││          │
   │  │  │ Paint event (triggered by Qt timer @ 30 Hz)  │││          │
   │  │  │ - Redraw reward time series (16 lines)       │││          │
   │  │  │ - Redraw contact map (4 lanes + gait shading)│││          │
   │  │  │ - Update env selector dropdown               │││          │
   │  │  └──────────────────────────────────────────────┘││          │
   │  │                                                  ││          │
   │  └──────────────────────────────────────────────────┘│          │
   │                                                       │          │
   │  ┌──────────────────────────────────────────────────┐│          │
   │  │ Qt main window                                   ││          │
   │  │  ┌─────────────────────┐  ┌──────────────────┐  ││          │
   │  │  │ Reward strip        │  │ Env selector     │  ││          │
   │  │  │ (pyqtgraph          │  │ (QComboBox)      │  ││          │
   │  │  │  PlotWidget, 16x)   │  └──────────────────┘  ││          │
   │  │  ├─────────────────────┤                        ││          │
   │  │  │ Contact strip       │  ┌──────────────────┐  ││          │
   │  │  │ (4 lanes + shading) │  │ Legend           │  ││          │
   │  │  │ + 3-leg gait marker │  │                  │  ││          │
   │  │  └─────────────────────┘  └──────────────────┘  ││          │
   │  │                                                  ││          │
   │  │ Status bar: fps, dropped_frames, last_msg_t     ││          │
   │  └──────────────────────────────────────────────────┘│          │
   │                                                       │          │
   └───────────────────────────────────────────────────────┘          │
```

### Data flow summary

1. **Sim process** publishes a snapshot of `[N=5, 16 rewards]`, `[N=5, 4 contact]`, step index, and timestamp to the ZMQ PUB socket **every physics step** (50 Hz).
2. **Viewer process** subscribes to this socket via a QSocketNotifier, appending each message to a ring buffer.
3. **Qt paint event** (on a 30 Hz timer) reads the ring buffer and renders the latest `env_selector`'s data slice.
4. **Scaling:** the viewer displays only the last 500 steps (≈10 seconds at 50 Hz), scrolling left-to-right.

---

## 3. IPC mechanism comparison table

| Mechanism | Pros | Cons | Latency | Throughput | Backpressure | Simplicity |
|-----------|------|------|---------|-----------|--------------|-----------|
| **ZeroMQ PUB/SUB** (ipc://) | Non-blocking pub; HWM auto-drops old frames; battle-tested; integrable with Qt via FD-based notifier; pyzmq is lightweight; no broker. | Requires `pyzmq` package; slightly opaque semantics around frame drops; small learning curve. | <1 ms (ipc) | >100k msg/s (plenty for 50 Hz) | HWM=10: publisher never blocks; SUB drops if lag >10 msgs. | High (standard pattern in telemetry). |
| **Unix domain socket + JSON** (AF_UNIX, SOCK_STREAM) | Standard POSIX; no external package (Python `socket` stdlib); human-readable payloads; easy to debug with `nc -U`. | Requires custom framing (length-prefix or delimiter); blocking reads by default (must use `fcntl.O_NONBLOCK`); no built-in pressure handling; single reader per socket. | <1 ms | Limited by syscall overhead (~10k msg/s) | Reader blocks if socket buffer fills (no recovery); data loss on discard. | Medium (more boilerplate). |
| **Shared memory** (multiprocessing.shared_memory or posix_ipc) | Lowest latency (kernel copy-on-write); zero-copy reads; very fast for large payloads. | Process coordination complexity (mutex/semaphore); platform-dependent (Linux/macOS only, no Windows); ring-buffer logic must be hand-written; harder to debug. | <0.1 ms | >1M msg/s (overkill) | Requires explicit mutex; deadlock risk if not careful. | Low (complex sync). |
| **Named pipe (FIFO)** | Simple syscall; buffered (PIPE_BUF ~4 KB). | Blocking by default; unidirectional only; small buffer; no backpressure semantics. | ~1 ms | Limited (~5k msg/s) | Writer blocks if buffer full; coarse backpressure. | Low–medium (dated pattern). |
| **UDP multicast** | Very fast; connectionless; good for many subscribers. | Unreliable (packet loss); requires network stack overhead; not suitable for local-machine single-process pairs. | ~1 ms | Excellent | None (UDP is lossy by design). | Medium (overkill). |

### Recommendation rationale

**ZeroMQ PUB/SUB is the clear winner** for this use case:

- **Non-blocking on publisher side:** The sim loop calls `pub_socket.send(msg)` in the hot path; if the viewer lags, the publisher does not block — it drops frames via HWM.
- **Seamless Qt integration:** `pyzmq.eventloop` provides a Qt notifier that wakes the event loop on message arrival, avoiding polling threads.
- **Proven in production telemetry:** ZeroMQ is used in robotics, HFT, and scientific simulation for exactly this pattern.
- **Latency/throughput sweet spot:** 50 Hz = 20 ms per frame; a 50 Mbps local IPC link handles 500 B/frame trivially.
- **Testability:** easy to mock a subscriber in unit tests or capture frames to disk for offline analysis.

**Why not shared memory:** the coordination overhead (semaphores, ring-buffer wraparound logic, inter-process synchronization) is not worth the minor latency win. At 50 Hz, <1 ms vs <0.1 ms is immaterial.

**Why not Unix socket + JSON:** it works, but requires custom framing and blocking-read handling. ZeroMQ abstracts these away and handles backpressure automatically.

---

## 4. Recommended stack

### Choice: ZeroMQ PUB/SUB + PyQt6 + pyqtgraph

**Data transport:** ZeroMQ `PUB/SUB` over ipc:// (Unix domain socket backend).

**GUI framework:** PyQt6 (or PySide6 if licensing preferred).

**Plotting:** **pyqtgraph** (not matplotlib).
- pyqtgraph is a GPU-accelerated plotting library built on OpenGL and designed for real-time updates.
- Ring-buffer updates at >1 kHz are routine; scrolling plots perform smoothly.
- Native integration with PyQt's paint pipeline (no blocking draws, no Qt event-loop collision).
- Lightweight dependency (wheels exist for all platforms).

**Message encoding:** msgpack (binary, compact, language-neutral).
- Compared to JSON: ~3× smaller payload (~150 B vs ~500 B for a step message).
- Compared to pickle: safer, deterministic, widely supported in robotics.
- At 50 Hz, 150 B/frame = 7.5 KB/s upstream; negligible.

### Rationale

1. **Performance:** pyqtgraph's GPU-accelerated plotting eliminates the GIL blocking problem that killed matplotlib. A pyqtgraph `PlotWidget` redraw is GPU-resident; the CPU is free for the Qt event loop.
2. **Isolation:** No collision with Isaac's embedded Qt. The viewer runs in its own Qt main loop in a separate Python process.
3. **Simplicity:** ZeroMQ PUB/SUB is fire-and-forget on the publisher side. The viewer's QSocketNotifier wakes the Qt event loop on message arrival — no polling thread needed.
4. **Debuggability:** msgpack is binary but can be inspected with `xxd` or a Python snippet. Easy to log and replay.

### Effort note

- **pyqtgraph** and **pyzmq** are not in the current `isaac-parkour` conda env.
- Both have wheels and can be `pip install`'d into the env without rebuilding from source.
- Footprint: ~50 MB combined (acceptable given a full Isaac Sim env is already 4 GB).

---

## 5. Data wire format

### Message schema (msgpack-encoded)

Each message sent by the sim process contains one timestamped snapshot of all active environments.

```python
# Pseudocode schema (not executable; for specification only)

Message = {
    "t": float64,                # Unix timestamp (seconds since epoch)
    "step_idx": uint32,          # Global step counter
    "envs": [
        {
            "env_id": uint8,     # 0–4 (which of the 5 parallel envs)
            "rewards": [float32] × 16,   # Scaled reward breakdown per term
            "contact": [bool] × 4,       # [FL, FR, RL, RR] foot contact
            "commands": [float32] × 3,   # [v_fwd, v_yaw, reserved]
            "done": bool,                # Episode termination flag
            "terrain_id": uint8,         # Class ID (0–4 for active terrains)
        },
        # ... up to 5 envs (only those not yet terminated in this run)
    ]
}
```

### Serialization example (pseudocode)

```python
import msgpack
import time
import numpy as np

# In the sim process, after env.step():
msg = {
    "t": time.time(),
    "step_idx": step_counter,
    "envs": [
        {
            "env_id": i,
            "rewards": env.unwrapped._last_reward_breakdown_per_env[i].cpu().numpy().tolist(),
            "contact": (torch.norm(forces[i, 0, feet_ids], dim=-1) > 2.0).cpu().numpy().tolist(),
            "commands": env.unwrapped._commands[i, :3].cpu().numpy().tolist(),
            "done": dones[i].item(),
            "terrain_id": env.unwrapped._env_class[i].item(),
        }
        for i in range(num_envs)
    ]
}

payload = msgpack.packb(msg, use_bin_type=True)  # ~150–200 bytes
pub_socket.send(payload)  # non-blocking
```

### Wire size estimate

- Base message (5 envs, all fields):
  - Metadata: t (8) + step_idx (4) + 5× env_id (5) = ~17 bytes
  - Data: 5 envs × (16 floats + 4 bools + 3 floats) = 5 × (64 + 1 + 12) = 385 bytes
  - msgpack overhead: ~15%
  - **Total: ~180–200 bytes per frame**

- At 50 Hz: 200 B/frame × 50 frames/s = 10 KB/s (**trivial over IPC**).

### Per-step vs per-batch decision

**Decision: per-step (as shown above).**

Rationale:
- The viewer's task is to correlate reward spikes with foot-contact phases in **real time**. Batching multiple steps into one message delays that correlation by the batch window.
- At 50 Hz, a single step costs negligible bandwidth (200 B/frame). No need to batch.
- Batching would complicate the ring-buffer logic in the viewer (variable message sizes).

---

## 6. Backpressure and frame-drop strategy

### Publisher side (sim process)

The ZeroMQ PUB socket has a **high-water-mark (HWM)** set to **10 messages**.

```python
# Sim-side setup (pseudocode)
pub_socket = zmq.PubSocket(...)
pub_socket.setsockopt(zmq.SNDHWM, 10)  # max 10 pending msgs before dropping
pub_socket.connect("ipc:///tmp/parkour_reward.sock")

# In the hot loop:
pub_socket.send(payload, flags=zmq.NOBLOCK)  # never blocks; drops if queue full
```

**Behavior:**
- If the viewer is slow and the ZMQ buffer fills to 10 messages, the **next `send()` call will silently drop the oldest message**.
- The publisher continues unimpeded; no backpressure, no stalls, no GIL contention.
- Dropped frames are acceptable here: we are visualizing a continuous signal; losing 1–2 frames at 50 Hz is imperceptible to the human eye (20–40 ms).

### Subscriber side (viewer process)

The viewer maintains a **ring buffer** of the last 500 time steps, which is ~10 seconds of wall-clock time at 50 Hz.

```python
# Viewer-side ring buffer (pseudocode)
class RingBuffer:
    def __init__(self, capacity=500, num_envs=5, num_terms=16):
        self.capacity = capacity
        self.rewards = np.zeros((capacity, num_envs, num_terms), dtype=np.float32)
        self.contact = np.zeros((capacity, num_envs, 4), dtype=bool)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.write_idx = 0
        self.size = 0
    
    def push(self, step_data):
        """Append one step; overwrites oldest if full."""
        self.rewards[self.write_idx] = step_data["rewards"]
        self.contact[self.write_idx] = step_data["contact"]
        self.timestamps[self.write_idx] = step_data["t"]
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_latest(self, env_id, lookback_steps=500):
        """Return the most recent `lookback_steps` for env_id."""
        if self.size == 0:
            return None
        # ... wrap-around logic ...
```

**Behavior:**
- Every message from the sim is appended to the ring buffer.
- If the viewer paints slower than the sim publishes (e.g., Qt tick at 30 Hz, sim at 50 Hz), the ring buffer will accumulate the difference. After ~16 steps, the oldest frame is overwritten.
- This is **expected and correct**: the viewer shows the most recent 500 steps, and older ones are discarded once the buffer fills.

### Observability

The viewer's status bar displays:
- **FPS (viewer paint rate):** updated every second.
- **Dropped frames estimate:** messages not received due to ZMQ HWM or viewer lag (tracked by comparing expected vs actual message IDs).
- **Last message age:** time since the most recent message arrived (alerts if >1 second = sim likely crashed).

---

## 7. Why this won't run inside the sim process

Explicit list of pitfalls for in-process visualization (why it was rejected):

### 7.1 Qt main loop collision
- **Problem:** Kit already owns a PySide2/PySide6 Qt event loop. Calling `QApplication.exec()` or spawning a second event loop in a background thread will cause a runtime error (Qt enforces single main event loop per process).
- **Attempted workaround:** running the Qt loop in a daemon thread; **does not work**. Qt's paint and input handling are not thread-safe; you get crashes or UI hangs.
- **Conclusion:** two Qt main loops = impossible in one process.

### 7.2 GIL and blocking draws
- **Problem:** matplotlib's `plt.draw()` and `Figure.canvas.draw_idle()` block for 5–20 ms per call on most backends. At 50 Hz (20 ms per step), a 10 ms draw = 50% of budget lost.
- **Attempted workaround:** deferring draws to a background thread; **does not work**. matplotlib is not thread-safe. Calling `draw()` from a non-main thread in most backends raises an error or hangs the GIL.
- **Attempted workaround:** using the `Agg` (non-interactive) backend + `Figure.savefig()` to a buffer; **works** but is even slower (requires PNG encoding).
- **Conclusion:** matplotlib in-process is unsuitable for real-time plotting in a GIL-heavy environment.

### 7.3 Policy loop latency impact
- **Problem:** even if draws are deferred, the policy step calls `_get_observations()`, `_get_rewards()`, `env.step()`, which must complete in <20 ms for a 50 Hz simulation. Any blocking I/O (socket recv), graphics (GPU command submission), or Python work (msgpack encoding) in that critical path will cause jitter.
- **In-process visualization forces the display code into the hot loop**, risking determinism.
- **Out-of-process visualization publishes data and immediately returns**, leaving the hot loop untouched.
- **Conclusion:** in-process visualization introduces latency risk; out-of-process is safer.

### 7.4 Isaac Sim renderpass integration
- **Problem:** Isaac Sim's physics loop and rendering loop are tightly coupled within the Omniverse Kit event loop. Embedding a competing graphics system (Qt or matplotlib) means managing two rendering pipelines simultaneously.
- **Potential issues:** VSync conflicts, double-buffering state inconsistency, shared GPU context management.
- **Conclusion:** safer to keep visualization in a separate process with its own GPU context.

### 7.5 Debugging and iteration
- **Problem:** if the visualization code crashes or hangs, the entire sim process crashes with it. Debugging is tied to the running simulation (no pause, replay, or independent investigation).
- **Out-of-process advantage:** a viewer crash does not stop the sim. You can stop the viewer, fix the code, and restart it independently.
- **Conclusion:** process isolation improves debuggability and iteration speed.

---

## 8. Implementation effort estimate

### Rough line-of-code breakdown

**Sim-side publisher hook** (~80 LOC):
- Import ZeroMQ, initialize PUB socket in viewer setup (~10 lines).
- Wrap reward/contact data into msgpack message (~20 lines).
- Publish on every step (~5 lines).
- Graceful shutdown (socket close) (~3 lines).

**Viewer application** (~400–500 LOC):
- Qt main window + layout (~80 lines).
- ZMQ SUB socket + QSocketNotifier (~50 lines).
- Ring buffer (custom or use `collections.deque`) (~60 lines).
- pyqtgraph PlotWidget setup (16-line reward strip + 4-lane contact) (~80 lines).
- Paint/redraw logic (color mapping, 3-leg gait shading) (~80 lines).
- Env selector (QComboBox) + event handlers (~30 lines).
- Status bar, legend, utility functions (~40 lines).

**Tests and utilities** (~50–80 LOC):
- Standalone ZMQ message generator (for unit tests without sim) (~20 lines).
- Viewer smoke test (recv N messages, check ring buffer) (~15 lines).
- Payload size verification script (~10 lines).

**Total: ~600–700 LOC across 2 scripts** (1 in the sim's scripts/ dir, 1 standalone viewer app).

### File structure

```
scripts/reinforcement_learning/rsl_rl/
├── play_reward_attribution_publisher.py  (80 LOC; hooks into main loop)
└── lib/
    └── reward_publisher.py               (utilities shared with publisher)

reward_attribution_viewer/
├── viewer.py                             (main Qt app; ~400 LOC)
├── ring_buffer.py                        (buffer logic; ~60 LOC)
├── message_pb2.py                        (msgpack schema helpers; ~20 LOC)
└── test_viewer.py                        (smoke test; ~30 LOC)
```

### Phase breakdown

| Phase | Work | Est. LOC | Timeline |
|-------|------|----------|----------|
| **1: Wire protocol** | Define msgpack schema, write standalone generator + unit tests. No graphics. | ~80 | 1–2 days (parallel: design verification) |
| **2: Publisher hook** | Integrate ZMQ PUB into play_reward_attribution.py after every step. Smoke test that messages flow. | ~80 | 1 day |
| **3: Viewer skeleton** | Qt main window, QSocketNotifier, ring buffer. Subscribe and log received messages. | ~150 | 2–3 days |
| **4: Plotting** | pyqtgraph PlotWidgets (reward + contact), color mapping, 3-leg gait shading, env selector. | ~200 | 3–4 days |
| **5: Polish** | Status bar, legend, keyboard shortcuts, save-to-PNG on demand, performance profiling. | ~100 | 2 days |
| **Total** | | ~610 LOC | 1–2 weeks (with parallelization) |

### Dependencies to add

Assuming the base `isaac-parkour` conda env is active:

```bash
pip install pyzmq pyqtgraph  # ~50 MB combined
```

No system-level dependencies (Qt6 is already available via pip on all platforms).

---

## 9. Alternative: Qt-in-thread and tornado-style fork

### Why in-thread Qt doesn't work

Some projects attempt to run a Qt event loop in a **background daemon thread** while the main thread holds the sim loop. This pattern is sometimes called "tornado-style" (borrowed from async frameworks that spin a separate event loop thread).

**Why it fails in Isaac Sim:**
1. **Qt is not thread-safe.** Calling `QWidget.update()`, `QApplication.processEvents()`, or GPU operations from a non-main thread results in crashes or undefined behavior.
2. **Python's GIL serializes access anyway.** Running the Qt loop in a background thread does not parallelize computation — both threads contend for the GIL. The sim thread may block while the Qt thread is executing, or vice versa.
3. **No isolation from Kit's event loop.** Kit's PySide loop is also in the main thread. Two event loops in the same thread = impossible.

### Why zmq-with-a-reader-thread fails

Some projects try:
- Main sim thread publishes to ZMQ PUB.
- Background reader thread subscribes, updates a shared data structure.
- Main sim thread (or yet another thread) renders using matplotlib or Qt.

**Why it fails:**
1. **Synchronization complexity.** Reader thread and render thread must coordinate via locks or queues. If either blocks, the main sim thread may be affected.
2. **Same Qt problem.** Rendering (Qt or matplotlib) is not thread-safe. You still can't render from a background thread.
3. **Negligible latency benefit.** At 50 Hz, the time from message arrival to redraw is dominated by the 30 ms Qt tick, not the microseconds saved by threading.

### Why separate processes is better

- **No GIL contention:** each process has its own Python interpreter and GIL.
- **Clean isolation:** if the viewer crashes, the sim keeps running.
- **Standard pattern:** process-based IPC is the de-facto solution in robotics, HFT, and scientific computing.
- **Easier testing:** you can mock the sim-side publisher or save/replay messages on disk.

---

## 10. Testing and validation checklist

Before the implementation phase, a reviewer should verify:

- [ ] All 9 sections of this spec are present and coherent.
- [ ] Wire format schema is concrete enough to write a unit test (section 5).
- [ ] Data flow diagram (section 2) matches the recommended stack (section 4).
- [ ] Effort estimate is broken into phases with clear deliverables (section 8).
- [ ] No unfeasible assumptions (e.g., assume Qt can run in-thread; it can't).
- [ ] Backpressure/drop strategy (section 6) is explicit and acceptable.

Once implementation begins:

- **Phase 1 validation:** standalone msgpack message generator produces messages that match the schema in section 5.
- **Phase 2 validation:** publish messages from a mock sim loop, verify no publisher-side stalls.
- **Phase 3 validation:** viewer receives and appends to ring buffer without blocking.
- **Phase 4 validation:** pyqtgraph redraw at 30 Hz without jitter; 3-leg gait shading appears on known test cases.
- **Phase 5 validation:** no segfaults, memory leaks, or frame drops over a 5-minute run.

---

## Appendix: Why section 8 (offline matplotlib) is preferred for MVP

The DESIGN.md stage-2 implementation uses **offline matplotlib** (not this PyQt+IPC design). Offline matplotlib:

- Runs *after* the episode is complete and the sim process has exited.
- No GIL or event loop collisions (no running sim).
- Can use standard `matplotlib.pyplot` without backend conflicts.
- Generates high-quality PNGs for archival.
- Slower to iterate (must re-run episode), but deterministic and debuggable.

**This PyQt+IPC spec is the recommended *stage-3+* evolution** if the user finds the offline plots insufficient for cause-finding. At that point, the `.npz` buffer format from DESIGN.md §2 becomes a live data source, and the viewer implementation above directly consumes it.

---

**End of specification.** Ready for stage-3+ implementation.
