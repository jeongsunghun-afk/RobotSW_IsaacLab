#!/usr/bin/env python3
"""Patch a copy of play.py -> play_flat_stream.py:
1. Force sim.render() every env step (mirror stance_view's render=True) so the
   WebRTC encoder actually receives frames (root cause: env.step gates rendering
   on has_gui()/has_rtx_sensors(), which is False under headless WebRTC livestream).
2. Print a one-line render diagnostic (has_gui / offscreen / render_mode) after the
   policy is built, so the log proves why the default gate was off.
"""
import io, sys, os

SRC = "/mnt/ssd1/jsh/RobotSW_IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"
DST = "/mnt/ssd1/jsh/RobotSW_IsaacLab/scripts/reinforcement_learning/rsl_rl/play_flat_stream.py"

with io.open(SRC, "r", encoding="utf-8") as f:
    s = f.read()

# --- Edit 1: diagnostic after the inference policy is obtained ---
anchor1 = "    policy = runner.get_inference_policy(device=env.unwrapped.device)\n    print(policy)\n"
assert anchor1 in s, "anchor1 not found"
diag = (
    "    policy = runner.get_inference_policy(device=env.unwrapped.device)\n"
    "    print(policy)\n"
    "    # ★[STREAMFIX] render diagnostic + force-render setup\n"
    "    _sc = env.unwrapped.sim\n"
    "    print(f\"[STREAMFIX] has_gui={_sc.has_gui()} offscreen={getattr(_sc,'_offscreen_render',None)} \"\n"
    "          f\"render_mode={_sc.render_mode}\", flush=True)\n"
)
s = s.replace(anchor1, diag, 1)

# --- Edit 2: force a render every env step, right after env.step() ---
anchor2 = "            obs, _, dones, _ = env.step(actions)\n"
assert anchor2 in s, "anchor2 not found"
forced = (
    "            obs, _, dones, _ = env.step(actions)\n"
    "            # ★[STREAMFIX] force a render each step so the livestream encoder gets frames.\n"
    "            #   env.step() only renders when has_gui()/has_rtx_sensors() is True, which is\n"
    "            #   False under headless WebRTC livestream -> encoder stays 0%. stance_view.py\n"
    "            #   worked because it called sim.step(render=True) explicitly. Do the same here.\n"
    "            env.unwrapped.sim.render()\n"
)
s = s.replace(anchor2, forced, 1)

with io.open(DST, "w", encoding="utf-8") as f:
    f.write(s)

print("WROTE", DST)
print("diag_present", "[STREAMFIX] render diagnostic" in s)
print("forced_present", "force a render each step" in s)
