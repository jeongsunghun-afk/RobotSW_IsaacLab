# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ZMQ + msgpack wire protocol for the live reward-attribution viewer.

Shared by **two** separate processes:
  - Sim-side publisher : ``play_reward_attribution.py --live-viz``  (Process A)
  - Viewer-side subscriber: ``reward_attribution_viewer.py``         (Process B)

Wire format overview
--------------------
The outer msgpack dict uses single-char keys to minimize overhead::

    {
        "v": int,  # PROTOCOL_VERSION (uint8)
        "t": float64,  # Unix timestamp (seconds since epoch)
        "s": uint32,  # Global step counter
        "n": uint8,  # Number of environments in this frame
        "e": [bytes, ...],  # Per-env binary blobs (one per env, see below)
    }

Each per-env binary blob is packed with :func:`struct.pack` in big-endian order::

    Offset   Size   Type        Field
    ------   ----   ----        -----
    0        1      uint8       env_id
    1        1      uint8       terrain_id
    2        1      uint8       contact_done byte:
                                   bits 0-3 = foot contact FL/FR/RL/RR (1=contact)
                                   bit  4   = episode done flag
                                   bits 5-7 = reserved (0)
    3        1      uint8       K  (number of reward terms, normally 16)
    4        2K     float16×K   per-term scaled reward breakdown (big-endian half-precision)
    4+2K     12     float32×3   command velocities [v_fwd, v_yaw, reserved]

For K=16: blob size = 4 + 32 + 12 = 48 bytes.
Full frame (N=5, K=16): outer (~22 B) + 5×(bin8_hdr(2)+blob(48)) = 272 B  [budget: <400 B].

Precision note
--------------
Rewards are downcast to **float16** (half-precision) before transmission.
float16 covers ±65 504 with ~3.3 significant decimal digits, which is more than
sufficient for plotting.  Typical parkour reward terms are in the range ±5 per
step; clamping to ±65504 is applied before packing to prevent struct overflow.

Design reference: PYQT_IPC_SPEC.md §4-§7
"""

from __future__ import annotations

import struct
import time
from typing import Any

import msgpack

# --------------------------------------------------------------------------- #
# Transport constants                                                           #
# --------------------------------------------------------------------------- #

ZMQ_ENDPOINT: str = "ipc:///tmp/parkour_reward.sock"
"""Unix-domain socket endpoint.  PUB (sim) binds; SUB (viewer) connects."""

ZMQ_HWM: int = 10
"""High-water-mark: publisher silently drops oldest frame when queue is full.
Prevents the sim hot-loop from blocking if the viewer falls behind.
Per PYQT_IPC_SPEC.md §6."""

PROTOCOL_VERSION: int = 1
"""Bump on any breaking schema change so the viewer can reject stale messages."""

# Internal struct formats (big-endian).
_HDR_FMT = "!BBBB"  # env_id, terrain_id, contact_done, K
_HDR_SIZE: int = struct.calcsize(_HDR_FMT)  # 4
_CMD_FMT = "!3f"  # 3 × float32 commands
_FLOAT16_MAX: float = 65504.0


# --------------------------------------------------------------------------- #
# Private helpers                                                               #
# --------------------------------------------------------------------------- #


def _pack_env(
    env_id: int,
    terrain_id: int,
    contact: list[bool],
    done: bool,
    rewards: list[float],
    commands: list[float],
) -> bytes:
    """Pack one environment's data into a fixed-layout binary blob."""
    k = len(rewards)
    contact_done_byte = (
        (int(bool(contact[0])))
        | (int(bool(contact[1])) << 1)
        | (int(bool(contact[2])) << 2)
        | (int(bool(contact[3])) << 3)
        | (int(bool(done)) << 4)
    )
    header = struct.pack(_HDR_FMT, env_id, terrain_id, contact_done_byte, k)
    # Clamp to float16 range before downcast to avoid struct overflow.
    rewards_clamped = [max(-_FLOAT16_MAX, min(_FLOAT16_MAX, float(r))) for r in rewards]
    rewards_bytes = struct.pack(f"!{k}e", *rewards_clamped)
    commands_bytes = struct.pack(_CMD_FMT, float(commands[0]), float(commands[1]), float(commands[2]))
    return header + rewards_bytes + commands_bytes


def _unpack_env(blob: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Unpack one per-env binary blob into a Python dict."""
    env_id, terrain_id, contact_done_byte, k = struct.unpack_from(_HDR_FMT, blob, 0)
    offset = _HDR_SIZE
    rewards: list[float] = list(struct.unpack_from(f"!{k}e", blob, offset))
    offset += k * 2
    commands: list[float] = list(struct.unpack_from(_CMD_FMT, blob, offset))

    contact: list[bool] = [(bool((contact_done_byte >> i) & 1)) for i in range(4)]
    done: bool = bool((contact_done_byte >> 4) & 1)

    return {
        "env_id": int(env_id),
        "terrain_id": int(terrain_id),
        "contact": contact,
        "done": done,
        "rewards": rewards,  # list[float], decoded from float16
        "commands": commands,
    }


# --------------------------------------------------------------------------- #
# Public API: encode / decode                                                  #
# --------------------------------------------------------------------------- #


def encode_step(
    step_idx: int,
    env_payloads: list[dict[str, Any]],
) -> bytes:
    """Encode one per-step snapshot to a compact msgpack + binary frame.

    Rewards are transmitted as float16 (half-precision) to keep the frame
    under 400 B for the nominal N=5, K=16 configuration.

    Args:
        step_idx:     Global step counter (wraps at 2³²).
        env_payloads: List of per-env dicts.  Required keys per entry:

            - ``env_id``    (int) – 0-based environment index.
            - ``rewards``   (list[float], len=K) – per-term scaled reward breakdown.
            - ``contact``   (list[bool], len=4) – foot contact [FL, FR, RL, RR].
            - ``commands``  (list[float], len=3) – [v_fwd, v_yaw, reserved].
            - ``done``      (bool) – episode termination flag.
            - ``terrain_id`` (int) – terrain class ID (0-based).

    Returns:
        msgpack-encoded bytes.
        Typical size: **≈ 274 B** for N=5, K=16 (well under the 400 B budget).
    """
    env_blobs = [
        _pack_env(
            e["env_id"],
            e["terrain_id"],
            e["contact"],
            e["done"],
            e["rewards"],
            e["commands"],
        )
        for e in env_payloads
    ]
    msg: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "t": time.time(),
        "s": step_idx,
        "n": len(env_payloads),
        "e": env_blobs,
    }
    return msgpack.packb(msg, use_bin_type=True)


def decode_step(payload: bytes) -> dict[str, Any]:
    """Decode a msgpack frame produced by :func:`encode_step`.

    Returns a dict with the following structure::

        {
            "v": int,  # protocol version
            "t": float,  # Unix timestamp
            "step_idx": int,  # global step counter
            "envs": [
                {
                    "env_id": int,
                    "terrain_id": int,
                    "contact": [bool, bool, bool, bool],
                    "done": bool,
                    "rewards": [float, ...],  # K values (decoded from float16)
                    "commands": [float, float, float],
                },
                ...,
            ],
        }

    Raises:
        ValueError: if the protocol version does not match :data:`PROTOCOL_VERSION`.
    """
    msg: dict[str, Any] = msgpack.unpackb(payload, raw=False)
    v = int(msg.get("v", 1))
    if v != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {v!r}; expected {PROTOCOL_VERSION}. "
            "Ensure sim and viewer use the same reward_pub_protocol.py."
        )
    return {
        "v": v,
        "t": float(msg["t"]),
        "step_idx": int(msg["s"]),
        "envs": [_unpack_env(blob) for blob in msg["e"]],
    }


# --------------------------------------------------------------------------- #
# Socket factories                                                              #
# --------------------------------------------------------------------------- #


def make_pub_socket(context: Any) -> Any:
    """Create a ZMQ PUB socket bound to :data:`ZMQ_ENDPOINT`.

    **Sim-process only.**  Binding means the endpoint persists as long as the
    sim runs; the viewer can connect and disconnect at any time.

    HWM = :data:`ZMQ_HWM` so a slow viewer never blocks the sim's hot loop
    (oldest queued message is silently dropped instead of stalling
    ``socket.send()``).  Per PYQT_IPC_SPEC.md §6.

    Args:
        context: An active ``zmq.Context`` instance.

    Returns:
        Configured, bound ZMQ PUB socket.
    """
    import zmq  # noqa: PLC0415 — deferred to keep import safe in minimal envs

    sock = context.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, ZMQ_HWM)
    sock.bind(ZMQ_ENDPOINT)
    return sock


def make_sub_socket(context: Any) -> Any:
    """Create a ZMQ SUB socket connected to :data:`ZMQ_ENDPOINT`.

    **Viewer-process only.**  Subscribes to all topics (empty prefix filter).
    RCVHWM is set to :data:`ZMQ_HWM` to bound the receive queue on the viewer
    side as well.

    Args:
        context: An active ``zmq.Context`` instance.

    Returns:
        Configured, connected ZMQ SUB socket.
    """
    import zmq  # noqa: PLC0415

    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, ZMQ_HWM)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # receive every message
    sock.connect(ZMQ_ENDPOINT)
    return sock


# --------------------------------------------------------------------------- #
# Self-test (importable + __main__)                                             #
# --------------------------------------------------------------------------- #


def _roundtrip_test(n_envs: int = 5, n_terms: int = 16, label: str = "") -> None:
    """Verify encode→decode round-trip, value fidelity, and the 400 B budget.

    Rewards are checked to within float16 precision (relative error ≤ 0.1 %).

    Raises:
        AssertionError: if any check fails.
    """
    import random

    prefix = f"[{label}] " if label else ""

    env_payloads = [
        {
            "env_id": i,
            "rewards": [random.gauss(0.0, 0.5) for _ in range(n_terms)],
            "contact": [random.random() > 0.5 for _ in range(4)],
            "commands": [1.0, 0.0, 0.0],
            "done": (i == 0),  # env 0 always done, rest not
            "terrain_id": i % 5,
        }
        for i in range(n_envs)
    ]

    payload = encode_step(step_idx=42, env_payloads=env_payloads)
    assert isinstance(payload, bytes), "encode_step must return bytes"

    frame_size = len(payload)
    assert frame_size < 400, (
        f"{prefix}Frame size {frame_size} B exceeds 400 B budget "
        f"(n_envs={n_envs}, n_terms={n_terms}). "
        "If K or N grew, check the wire format notes at the top of this module."
    )

    decoded = decode_step(payload)
    assert decoded["step_idx"] == 42, f"{prefix}step_idx round-trip mismatch"
    assert len(decoded["envs"]) == n_envs, f"{prefix}envs count: got {len(decoded['envs'])}, want {n_envs}"

    for i, env in enumerate(decoded["envs"]):
        orig = env_payloads[i]
        assert env["env_id"] == orig["env_id"], f"{prefix}env_id mismatch at {i}"
        assert env["terrain_id"] == orig["terrain_id"], f"{prefix}terrain_id mismatch at {i}"
        assert env["done"] == orig["done"], f"{prefix}done mismatch at env {i}"
        assert env["contact"] == orig["contact"], f"{prefix}contact mismatch at env {i}"
        assert len(env["rewards"]) == n_terms, f"{prefix}rewards length mismatch at env {i}"
        # Verify float16 precision: relative error ≤ 0.2 % for values in ±5 range.
        for j, (got, want) in enumerate(zip(env["rewards"], orig["rewards"])):
            abs_err = abs(got - want)
            if abs(want) > 1e-6:
                rel_err = abs_err / abs(want)
                assert rel_err < 0.002, (
                    f"{prefix}reward[{i}][{j}]: {want:.6f} → {got:.6f} (rel_err={rel_err:.4f} > 0.002)"
                )

    print(
        f"{prefix}round-trip OK  "
        f"frame={frame_size} B  budget_used={100 * frame_size // 400}%  "
        f"n_envs={n_envs}  n_terms={n_terms}"
    )


if __name__ == "__main__":
    _roundtrip_test(n_envs=5, n_terms=16, label="nominal N=5 K=16")
    _roundtrip_test(n_envs=5, n_terms=20, label="guard  N=5 K=20")
    _roundtrip_test(n_envs=1, n_terms=16, label="single N=1 K=16")
    print("\n[reward_pub_protocol] all self-tests passed.")
