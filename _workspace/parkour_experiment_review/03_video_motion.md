# 03 — Parkour Experiment Video & Motion Quality Review

Owner: worker-video • Generated: 2026-05-14 • Source: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/<exp>/videos/train/`

ffmpeg note: system `ffmpeg` was not in `PATH`. Fell back to `/home/user/miniconda3/envs/vlnce/bin/ffmpeg` (v9c33b2f, gcc 9.3.0). All extractions used `-ss <t> -frames:v 1 -y`. Frames saved to `frames/<exp>/frame_XX.png` (5 frames @ 0%, 25%, 50%, 75%, 100% of duration).

---

## Section 1 — All 24 Experiments: Last-Video Metadata

Sorted chronologically (by directory timestamp). `last_step` is the highest numeric step suffix in `videos/train/rl-video-step-N.mp4`.

| # | Experiment | #videos | last_step | last file (bytes) | has_video |
|---|---|---:|---:|---:|:---:|
| 1 | 2026-05-11_15-27-00_change_actuator | 233 | 98000 | 9,470,381 | ✅ |
| 2 | 2026-05-11_16-25-20_isaaclab_actuator | 518 | 998000 | 6,358,789 | ✅ |
| 3 | 2026-05-11_17-27-46_change_learning_rate | 275 | 98000 | 13,250,982 | ✅ |
| 4 | 2026-05-11_17-46-01_damping_2.0 | 183 | 98000 | 5,920,300 | ✅ |
| 5 | 2026-05-11_17-47-11_add_feet_dragging | 185 | 98000 | 8,843,011 | ✅ |
| 6 | 2026-05-12_09-34-30_add_feet_dragging_0.1 | 40 | 8000 | 9,747,108 | ✅ |
| 7 | 2026-05-12_13-01-36_add_feet_dragging_0.1 | 37 | 8000 | 8,899,294 | ✅ |
| 8 | 2026-05-12_13-05-51_damping_1.0 | 35 | 8000 | 9,992,704 | ✅ |
| 9 | 2026-05-12_18-23-24_change_various_thing_yaw_diff | 188 | 98000 | 7,156,887 | ✅ |
| 10 | 2026-05-12_18-23-51_change_various_things_0.5_yaw_diff | 224 | 98000 | 11,760,143 | ✅ |
| 11 | 2026-05-13_09-28-12_flat_terrain | 38 | 8000 | 12,251,670 | ✅ |
| 12 | 2026-05-13_12-26-46_flat_terrain_yaw_wrap | 0 | — | 0 | ❌ |
| 13 | 2026-05-13_12-30-44_flat_terrain_yaw_wrap_dof_error | 0 | — | 0 | ❌ |
| 14 | 2026-05-13_12-39-19_flat_terrain_change | 10 | 8000 | 18,726,974 | ✅ |
| 15 | 2026-05-13_13-37-21_flat_terrain_change | 64 | 98000 | 9,380,977 | ✅ |
| 16 | 2026-05-13_14-10-44 | 404 | 99000 | 13,225,710 | ✅ |
| 17 | 2026-05-13_14-46-37_action_rate_0.1 | 41 | 8000 | 11,586,054 | ✅ |
| 18 | 2026-05-13_18-10-09_except_foot_penalty | 184 | 98000 | 8,250,607 | ✅ |
| 19 | 2026-05-13_18-32-36_except_foot_friction_average | 182 | 98000 | 12,555,373 | ✅ |
| 20 | **2026-05-14_09-23-04_except_foot_penalty_tracking_1.0** ⭐ | 33 | 8000 | 10,011,016 | ✅ |
| 21 | **2026-05-14_09-26-09_up_no_dof_error** ⭐ | 54 | 9000 | 8,681,000 | ✅ |
| 22 | **2026-05-14_09-26-42_except_foot_friction_average_no_action_penalty** ⭐ | 26 | 8000 | 10,093,974 | ✅ |
| 23 | **2026-05-14_10-27-11_change_collision** ⭐ | 18 | 8000 | 6,815,145 | ✅ |
| 24 | **2026-05-14_12-12-15_change_physx_config** ⭐ | 2 | 2000 | 16,270,484 | ✅ |

⭐ = top-5 most recent experiments visually reviewed in Section 2.

Observations on training duration:
- Two experiments produced zero videos (#12, #13 — yaw_wrap runs), likely crashed/aborted early.
- The top-5 most recent runs are all short (≤9000 steps; #24 only 2000 steps), so the visual review below reflects **very early-training behavior**, not converged policies.
- Long runs (98k–998k steps) clustered in 2026-05-11 → 2026-05-13_18. Run #2 (998k) is by far the longest.

---

## Section 2 — Top-5 Latest Experiments: Frame-Level Motion Review

All five videos are ~20 s long. Five frames per video were extracted at t ≈ 0 / 5 / 10 / 15 / 20 s. Every observation below comes from direct visual inspection of the PNGs in `frames/<exp>/`. **Important caveat**: ≤9k training steps is *very* early in PPO training — these are not converged behaviors and should not be read as "final policy quality."

### Exp #20 — `2026-05-14_09-23-04_except_foot_penalty_tracking_1.0` (last_step=8000, 33 videos)

| Frame | Visual description |
|---|---|
| 00 (t≈0s) | Two foreground robots clearly standing on 4 legs with legs roughly under the body, body height looks normal. Rows of robots in mid/far-field arranged on tiled terrain (step/stair tiles visible as grid). Background row shows several robots in similar upright stance. |
| 01 (t≈5s) | Foreground robot in center on 4 legs but with a slight forward lean; another adjacent robot has legs splayed wider than natural Go2 width. Background rows still mostly upright, some scatter. |
| 02 (t≈10s) | Single foreground robot in clean upright trotting/standing pose — limbs roughly tucked under, body level. Background mass orderly. Best-looking frame of the sequence. |
| 03 (t≈15s) | Center-front robot in a low-crouch with legs spread wide (more "frog-like" splay than canonical Go2 stance). Adjacent robot looks more normal. Mid-field mostly upright. |
| 04 (t≈20s) | Two foreground robots with legs splayed wide and hips noticeably low; one looks asymmetric (twisted). Background row uniform and upright. |

**Overall judgement**: Robots are *standing* (no clear falls) but the foreground stance is **inconsistently splayed / low-hipped** — not a confident parkour gait. No clear terrain-traversal progress observable in the cropped view. Yaw appears stable. Reads as "balancing but not striding."

---

### Exp #21 — `2026-05-14_09-26-09_up_no_dof_error` (last_step=9000, 54 videos)

| Frame | Visual description |
|---|---|
| 00 (t≈0s) | Foreground robots upright on 4 legs, body height normal, legs reasonably under torso. Rows of robots in mid-field arranged orderly. |
| 01 (t≈5s) | Front-right robot caught mid-stride with one leg lifted (looks like a real step). Body level. Other robots showing varied stride phases — suggests asynchronous walking rather than freezing. |
| 02 (t≈10s) | Foreground pair in a compact stance, very close together — possibly approaching same terrain feature. Body posture upright. |
| 03 (t≈15s) | Front robots in low-stance but legs tucked (not splayed) — body height a bit reduced but symmetric, not collapsed. |
| 04 (t≈20s) | Foreground robot in clean trot pose, legs in diagonal-pair phase typical of trotting gait. Body level. |

**Overall judgement**: **Visually the cleanest of the five.** Legs stay tucked under the body, body height looks stable, and several frames catch obvious mid-stride poses suggesting actual locomotion (not just balancing). Yaw / pitch / roll all look stable. This is the most "Go2-like" gait pose among the top-5.

---

### Exp #22 — `2026-05-14_09-26-42_except_foot_friction_average_no_action_penalty` (last_step=8000, 26 videos)

| Frame | Visual description |
|---|---|
| 00 (t≈0s) | Three foreground robots, body upright but **legs noticeably wide-set** (splayed). Background rows orderly. |
| 01 (t≈5s) | Front-left robot in low-hipped pose; the cluster shows mostly upright but with wide stance. Some asymmetry visible (one back leg further out than the other). |
| 02 (t≈10s) | Two foreground robots: front one with hips dropped very low, back legs spread wide; second one upright. Looks like the policy oscillates between crouch and stand. |
| 03 (t≈15s) | Foreground robot in canonical 4-leg stance, body level. Better posture than frame 02. |
| 04 (t≈20s) | Two foreground robots: left one slightly tilted (mild roll), right one upright. Both standing. |

**Overall judgement**: Similar splayed/low-hip pattern to Exp #20 but with marginally better recovery moments. Robots are *not* falling, but the gait alternates between healthy stance and over-splayed crouch — the missing action penalty plausibly correlates with the wider, more "loose" leg motion. No clear sustained forward locomotion observed in this 20s slice.

---

### Exp #23 — `2026-05-14_10-27-11_change_collision` (last_step=8000, 18 videos)

| Frame | Visual description |
|---|---|
| 00 (t≈0s) | **Front-left robot appears collapsed / nearly flat on ground** — body very close to ground, splayed legs. Adjacent robot also low-bodied. Mid-field robots upright. |
| 01 (t≈5s) | A single isolated robot in foreground, clearly trotting forward with one foreleg lifted. Body upright, clean pose. Background mass present but distant. |
| 02 (t≈10s) | One foreground robot in clean walking pose, body level, legs tucked. Mid-field rows upright. |
| 03 (t≈15s) | Two foreground robots side-by-side, both upright, body height normal. Background rows orderly. |
| 04 (t≈20s) | Foreground pair walking, one with foreleg lifted mid-stride. Body level. |

**Overall judgement**: One ambiguous near-fallen pose in frame 00, but frames 01–04 show **upright walking with visible stride phases**. The single isolated foreground robot in frame 01–02 suggests the camera is tracking one env and that env survived. Among the 8k-step short runs, this looks reasonably healthy — comparable to Exp #21.

---

### Exp #24 — `2026-05-14_12-12-15_change_physx_config` (last_step=2000, 2 videos) ⚠️ very early

| Frame | Visual description |
|---|---|
| 00 (t≈0s) | Foreground row of robots with hips dropped low and legs splayed extremely wide — bodies sit almost on the ground. Some robots appear face-down. |
| 01 (t≈5s) | **Many foreground robots collapsed / face-planted** — body nose-down on tiles, legs sprawled. Looks like a mass fall. |
| 02 (t≈10s) | Foreground robots in disorderly low/collapsed cluster — bodies on ground, multiple legs splayed. |
| 03 (t≈15s) | Wide foreground row, all robots in very low / collapsed stance, dense overlap of legs — robots not separated into walking columns. |
| 04 (t≈20s) | Disordered foreground cluster, low-bodied / splayed. Background row of robots more upright by comparison. |

**Overall judgement**: **Clearly pre-locomotion.** At 2000 PPO steps the policy has not yet learned to stand reliably — most foreground envs are collapsed or face-down. This is *not* indicative of the physx_config change being bad; the run simply hasn't had time to train. Compare to Exp #21 at 9000 steps which already shows striding behavior.

---

## Section 3 — Qualitative Motion-Quality Ranking (Top-5 Only)

⚠️ **This is a qualitative, single-rater visual ranking from 5 sparse frames per run.** It says nothing about converged performance; all top-5 runs are <10k steps. Use only as a rough early-training behavioral fingerprint, never as a reward-quality verdict.

| Rank | Experiment | last_step | Why (frame-evidence only) |
|---|---|---:|---|
| 🥇 1 | `2026-05-14_09-26-09_up_no_dof_error` | 9000 | Cleanest stances; legs tucked under body; multiple frames catch obvious mid-stride poses; no visible falls. |
| 🥈 2 | `2026-05-14_10-27-11_change_collision` | 8000 | 4 of 5 frames show upright walking with visible stride phases; one ambiguous low-body pose in frame 0. |
| 🥉 3 | `2026-05-14_09-23-04_except_foot_penalty_tracking_1.0` | 8000 | Robots all upright (no falls) but stance often splayed/low-hipped — balancing more than striding. |
| 4 | `2026-05-14_09-26-42_except_foot_friction_average_no_action_penalty` | 8000 | Similar splay/crouch pattern to #3, with mild posture oscillation. Slightly looser stance plausibly tied to dropped action penalty. |
| 5 | `2026-05-14_12-12-15_change_physx_config` | 2000 | Robots mostly collapsed / face-down — but **run is only 2k steps**, so this ranking is unfair to the config change. Needs more training to be comparable. |

### Cross-cutting notes for team-lead

1. **Top-5 are all very early-training runs** (≤9k steps). The clearest "good gait" signals are upright stance + visible stride asynchrony — both present in #21 and #23, absent in #24.
2. The splayed/low-hip pattern in #20/#22 may correlate with reward changes around foot/friction terms; worth checking against worker-params' diff.
3. #24 cannot be fairly compared — at 2000 steps the policy is in its pre-locomotion phase. **Recommend a follow-up review once it reaches ~8–10k steps.**
4. No video shows clear traversal across step/stair/gap terrain features in the visible foreground — likely because all top-5 are too early to be testing terrain progression. Long-run experiments (#2 at 998k, #16 at 99k) were not visually reviewed in this pass but would show converged behavior.
5. ffmpeg-availability: a system-level `ffmpeg` install is recommended; relying on the `vlnce` conda env binary is fragile.
