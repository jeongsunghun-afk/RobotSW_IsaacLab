import io, ast
p = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
s = io.open(p, encoding="utf-8").read()

# ── 1) 사인파 해석해를 제거하고 "캐시 위상평균"으로 교체 ────────────────────────
old_start = '            if getattr(self, "_sref_analytic", False):'
i = s.index(old_start)
j = s.index('            _yaw = torch.atan2(', i)
new_block = '''            if getattr(self, "_sref_analytic", False):
                # [SREF-S'] 몸통참조 = 같은 캐시의 **위상평균** 계획.  (phase_x, phase_y) 로 평균내
                # **지형 조건화만** 제거하고 (vx, level) 조건은 남긴다.  앞선 팔S 는 sway 위상/모양을
                # 내가 추론으로 정했다가 캐시 실측과 부호·로브수가 반대였다(캐시=−y 단일로브) →
                # 스윙 다리 쪽으로 몸을 기울여 정책이 STALL 을 택했다.  여기서는 추측을 전부 제거한다.
                plan = self._sref_br_mean[vx, lvl]  # (N,n_t,6)
'''
s = s[:i] + new_block + s[j:]

# ── 2) 위상평균 테이블 생성(캐시 로드 직후, solve_ok 가중) ─────────────────────
a = '''        self._sref_ok_tab = torch.tensor(
            ok.reshape(nvx, nlv, npx, npy), device=self.device, dtype=torch.float)'''
assert s.count(a) == 1, ("oktab", s.count(a))
s = s.replace(a, a + '''
        # [SREF-S'] 위상평균 계획 (vx, level, n_t, 6) — solve_ok 셀만 가중평균.
        _w = self._sref_ok_tab[..., None, None]                       # (vx,lv,px,py,1,1)
        _den = _w.sum(dim=(2, 3)).clamp(min=1e-6)                     # (vx,lv,1,1)
        self._sref_br_mean = (self._sref_br_tab * _w).sum(dim=(2, 3)) / _den''', 1)

# ── 3) 배너 — 다른 게이트와 같은 규약(소스를 로그로 증명 가능하게) ─────────────
b = '''            f"fh{tuple(self._sref_fh_tab.shape)} baseref{tuple(self._sref_br_tab.shape)} | "'''
assert s.count(b) == 1, ("banner", s.count(b))
s = s.replace(b, b + '''
            f"BASEREF_SRC={'cycle-mean(phase-averaged)' if getattr(self, '_sref_analytic', False) else 'TAMOLS(terrain-conditioned)'} | "''', 1)

# ── 4) 이제 안 쓰는 진폭 게이트 정리 ──────────────────────────────────────────
s = s.replace('        self._sref_sway_amp = float(getattr(self.cfg, "sref_sway_amp", 0.04))\n', '', 1)

ast.parse(s)
io.open(p, "w", encoding="utf-8").write(s)
print("팔 S' 적용 4/4 — 위상평균 계획 + BASEREF_SRC 배너 · 문법 OK")
