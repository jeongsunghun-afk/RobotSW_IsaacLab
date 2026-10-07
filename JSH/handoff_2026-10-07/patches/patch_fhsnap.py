import io, ast
p = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
s = io.open(p, encoding="utf-8").read()

a = '        cfg.step_tamols_fh = os.environ.get("GO2_STEP_TAMOLS_FH", "").strip() == "1"'
assert s.count(a) == 1, ("gate", s.count(a))
s = s.replace(a, a + '\n'
 '        # [SREF] TAMOLS 발판 xy 에도 기하경로와 **동일한 최근접-돌 스냅**을 적용한다.\n'
 '        # 왜: 팔G(기하)=lookahead+스냅 / 팔T(TAMOLS)=스냅없음 이라 비교에 스냅이 제2변수로 섞였다.\n'
 '        # 이 게이트를 켜면 단일변수가 "배치의 출처(TAMOLS vs Raibert)" 하나로 좁혀진다.\n'
 '        cfg.step_tamols_fh_snap = os.environ.get("GO2_STEP_TAMOLS_FH_SNAP", "").strip() == "1"', 1)

b = '        self._sref_fh = bool(getattr(self.cfg, "step_tamols_fh", False))'
assert s.count(b) == 1, ("flag", s.count(b))
s = s.replace(b, b + '\n        self._sref_fh_snap = bool(getattr(self.cfg, "step_tamols_fh_snap", False))', 1)

c = '                self._snap_xy_to_stone(tgt[..., :2])\n                tgt[..., 2] = self._snap_topz'
assert s.count(c) == 1, ("snap", s.count(c))
s = s.replace(c,
 '                _snapped_xy = self._snap_xy_to_stone(tgt[..., :2])\n'
 '                if getattr(self, "_sref_fh_snap", False):\n'
 '                    tgt[..., :2] = _snapped_xy   # [SREF-SNAP] 기하경로와 동일하게 돌 안으로 클램프\n'
 '                tgt[..., 2] = self._snap_topz', 1)

d = "f\"FOOTHOLD_SRC={'TAMOLS' if self._sref_fh else 'geometric-snap(unchanged)'} \""
assert s.count(d) == 1, ("banner", s.count(d))
s = s.replace(d, "f\"FOOTHOLD_SRC={('TAMOLS+snap' if self._sref_fh_snap else 'TAMOLS') if self._sref_fh else 'geometric-snap(unchanged)'} \"", 1)

ast.parse(s)
io.open(p, "w", encoding="utf-8").write(s)
print("GO2_STEP_TAMOLS_FH_SNAP 게이트 추가 4/4 · 문법 OK · 미설정 시 동작 불변")
