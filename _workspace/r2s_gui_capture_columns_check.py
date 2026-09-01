# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""gui_controller 정책 캡처 열 검정 — Qt/Isaac 없이 소스 텍스트를 직접 검정한다.

대상: 2026-09-01 에 추가한 raw_action(정책 원본 출력) · tau_sim/ddq_sim(sim 리치 채널) ·
버려지고 있던 TELEM 세 열.
"""
import ast, io, os, re, sys, tempfile, textwrap
import numpy as np

P = "scripts/real2sim/r2s_biped_leg/gui_controller.py"
src = io.open(P, encoding="utf-8").read()
tree = ast.parse(src)
ok = True


def chk(name, cond, extra=""):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


# ── 1) 공유메모리 레이아웃이 겹치지 않는가 ────────────────────────────────────────
C = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
        nm = n.targets[0].id
        if nm.startswith(("_SM_", "MAX_FRAMES", "NUM_JOINTS")) and isinstance(n.value, ast.Constant):
            C[nm] = n.value.value
C.setdefault("NUM_JOINTS", 8)
NJ = 8
blocks = [("_SM_SIM_TAU", NJ), ("_SM_SIM_DDQ", NJ), ("_SM_SIM_RICH_T", 1)]
occupied = set()
for nm, w in blocks:
    rng = set(range(C[nm], C[nm] + w))
    chk(f"{nm} 슬롯 미충돌", not (rng & occupied), f"{C[nm]}..{C[nm]+w-1}")
    occupied |= rng
chk("_SM_FRAMES 가 새 블록 뒤에 온다", C["_SM_FRAMES"] > max(occupied), f"_SM_FRAMES={C['_SM_FRAMES']}")
chk("기존 _SM_CHIRP_CENTER 블록과 미충돌", not (set(range(C["_SM_CHIRP_CENTER"], C["_SM_CHIRP_CENTER"] + NJ)) & occupied))

# ── 2) publisher 가 쓰는 슬롯과 정책 스레드가 읽는 슬롯이 같은가 ──────────────────
wr = set(re.findall(r"shared\[(_SM_SIM_\w+)", src))
rd = set(re.findall(r"self\._shared\[(_SM_SIM_\w+)", src))
chk("publisher 기록 ↔ 정책 스레드 판독 슬롯 일치", wr == rd == {"_SM_SIM_TAU", "_SM_SIM_DDQ", "_SM_SIM_RICH_T"},
    f"쓰기 {sorted(wr)} / 읽기 {sorted(rd)}")

# ── 3) leg-major → articulation 재배열이 맞는가 ──────────────────────────────────
LM = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
ART = ["HL_hip", "HR_hip", "HL_thigh", "HR_thigh", "HL_calf", "HR_calf", "HL_foot", "HR_foot"]
lm_for_art = next(
    [e.value for e in n.value.elts]
    for n in tree.body
    if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_LM_FOR_ART"
)
chk("_LM_FOR_ART 재배열이 이름 기준으로 맞는다", [LM[lm_for_art[p]] for p in range(NJ)] == ART)

# ── 4) 저장 함수 실행 — 새 열이 실제로 npz 에 들어가는가 ──────────────────────────
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_save_policy_recording")
out_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(out_dir, "data", "bipedleg_gui"), exist_ok=True)


class _Lbl:
    def setText(self, s):
        self.txt = s


class _Self:
    _status_label = _Lbl()


ns = dict(np=np, os=os, time=__import__("time"), NUM_JOINTS=NJ, ACTION_SCALE=0.25,
          DEFAULT_JOINT_POS=[0.0] * NJ, _REPO_ROOT=out_dir)
exec(compile(textwrap.dedent(ast.get_source_segment(src, fn)), P, "exec"), ns)
save = ns["_save_policy_recording"]

rng_ = np.random.default_rng(1)
rows, exp_tau, exp_ddq, stale = [], [], [], []
for i in range(120):
    raw = rng_.uniform(-3, 3, NJ).astype(np.float32)
    tau = rng_.uniform(-9, 9, NJ).astype(np.float32)
    ddq = rng_.uniform(-40, 40, NJ).astype(np.float32)
    is_stale = i % 30 == 29  # 이 틱만 sim 리치가 묵어 None 이어야 한다
    rows.append((float(i), i, 0.0, 0.0, 0.0, 0.5, [0.0] * NJ, [0.0] * NJ,
                 [0.0] * NJ, [0.0] * NJ, [0, 0, -1.0], [0.0] * NJ, [0.0] * NJ, [0, 0, -1.0],
                 [], [], [0.0] * NJ, list(raw), list(raw),
                 None if is_stale else list(tau), None if is_stale else list(ddq)))
    exp_tau.append(tau); exp_ddq.append(ddq); stale.append(is_stale)
telem = [(float(i) * 0.005, [0.1] * NJ, [0.2] * NJ, [0.3] * NJ, i, [0.4] * NJ, (1 << 16) | i % 7)
         for i in range(50)]
save(_Self(), rows, telem)
f = sorted(os.listdir(os.path.join(out_dir, "data", "bipedleg_gui")))[-1]
d = np.load(os.path.join(out_dir, "data", "bipedleg_gui", f))
st = np.asarray(stale)

chk("새 열 존재", {"tau_sim", "ddq_sim", "raw_action_sim", "raw_action_real"} <= set(d.files))
chk("tau_sim 값 일치(신선 틱)", np.allclose(d["tau_sim"][~st], np.asarray(exp_tau)[~st], atol=1e-5))
chk("ddq_sim 값 일치(신선 틱)", np.allclose(d["ddq_sim"][~st], np.asarray(exp_ddq)[~st], atol=1e-4))
chk("묵은 틱은 NaN", np.isnan(d["tau_sim"][st]).all() and np.isnan(d["ddq_sim"][st]).all(), f"({st.sum()} 틱)")
chk("행 수 보존(묵은 틱을 버리지 않는다)", len(d["t"]) == len(rows))
chk("TELEM 세 열 복구", {"telem_tick", "telem_cmd_q_lm", "telem_clamp_mask"} <= set(d.files))
chk("telem_tick 단조", np.all(np.diff(d["telem_tick"]) == 1))
chk("telem_clamp_mask 보존", int(d["telem_clamp_mask"][3]) == ((1 << 16) | 3))
chk("기존 열 불변", d["target_real"].shape == (120, NJ) and d["clamped_real"].shape == (120, NJ))

print("\n전 항목 통과" if ok else "\n실패 있음")
sys.exit(0 if ok else 1)
