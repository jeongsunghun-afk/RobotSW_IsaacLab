import io, ast
p = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
s = io.open(p, encoding="utf-8").read()

# 1) 게이트
a = '        cfg.step_tamols_fh_snap = os.environ.get("GO2_STEP_TAMOLS_FH_SNAP", "").strip() == "1"'
assert s.count(a) == 1, ("gate", s.count(a))
s = s.replace(a, a + '\n'
 '        # [SREF-S] 몸통참조를 TAMOLS 캐시가 아니라 **해석해**로 만든다(팔 S).\n'
 '        #   x = vx_cmd 등속 · z = 등고(캐시 실측 평균 0.32) · y = 게이트 동기 사인파 sway.\n'
 '        #   캐시 계획의 sway 는 셀당 8cm 인데 전 셀 평균은 2.7cm 다 = sway 대부분이 게이트가 아니라\n'
 '        #   **지형/상태에 조건화**돼 있다는 뜻.  팔 S 는 게이트 고정 성분만 남기고 그 조건화를 제거해\n'
 '        #   "최적화된 모양이 필요한가, 올바른 위상의 sway 면 충분한가"를 가른다.\n'
 '        cfg.sref_base_analytic = os.environ.get("GO2_SREF_BASE_ANALYTIC", "").strip() == "1"\n'
 '        cfg.sref_sway_amp = float(os.environ.get("GO2_SREF_SWAY_AMP", "0.04"))   # 진폭(±) → 8cm p-p', 1)

# 2) 런타임 플래그
b = '        self._sref_fh_snap = bool(getattr(self.cfg, "step_tamols_fh_snap", False))'
assert s.count(b) == 1, ("flag", s.count(b))
s = s.replace(b, b + '\n'
 '        self._sref_analytic = bool(getattr(self.cfg, "sref_base_analytic", False))\n'
 '        self._sref_sway_amp = float(getattr(self.cfg, "sref_sway_amp", 0.04))', 1)

# 3) 재앵커 시 계획을 해석해로 대체
c = '            plan = self._sref_br_tab[vx, lvl, ix, iy]  # (N,n_t,6)'
assert s.count(c) == 1, ("plan", s.count(c))
s = s.replace(c, c + '\n'
 '            if getattr(self, "_sref_analytic", False):\n'
 '                # (N,n_t,6) local = [x, y, z_rel_ground, vx, vy, vz]\n'
 '                _nt = self._sref_nt\n'
 '                _T = 4.0 * 0.2                                    # walk 4위상 × 0.2s = 캐시 호라이즌\n'
 '                _tau = torch.linspace(0.0, _T, _nt, device=self.device)[None, :]   # (1,nt)\n'
 '                _vx = self._commands[:, 0:1].clamp(min=0.0)       # 명령 전진속도 (N,1)\n'
 '                _A = self._sref_sway_amp\n'
 '                _w = 2.0 * 3.141592653589793 / _T\n'
 '                _zc = torch.full_like(_tau.expand(self.num_envs, -1), 0.32)  # 캐시 실측 평균 높이\n'
 '                plan = torch.stack([\n'
 '                    _vx * _tau,                                   # x\n'
 '                    _A * torch.sin(_w * _tau).expand(self.num_envs, -1),        # y sway\n'
 '                    _zc,                                          # z (지면 기준)\n'
 '                    _vx.expand(-1, _nt),                          # vx\n'
 '                    (_A * _w) * torch.cos(_w * _tau).expand(self.num_envs, -1), # vy\n'
 '                    torch.zeros_like(_zc),                        # vz\n'
 '                ], dim=2)', 1)

ast.parse(s)
io.open(p, "w", encoding="utf-8").write(s)
print("팔 S 게이트 GO2_SREF_BASE_ANALYTIC 추가 3/3 · 문법 OK · 미설정 시 동작 불변")
