import io
p = "source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
s = io.open(p, encoding="utf-8").read()

# ---- (1) comment block: the per-foot number is phi; touchdown is at t = 1 - phi ----
old = (
    "    #   trot  phase=0.5 offset=0   bound=0   -> FL,RR @0.5 ; FR,RL @0   (diagonal pairs)\n"
    "    #   walk  phase=0   offset=.25 bound=.5  -> RR@0, FR@.25, RL@.5, FL@.75 (4-beat lateral seq)\n"
)
assert s.count(old) == 1, "c1"
new = (
    "    # NOTE the per-foot number above is that foot PHASE OFFSET phi; the foot TOUCHES DOWN at\n"
    "    # t = 1 - phi (stance occupies [phi, phi+duty) of the warped clock), so a LARGER phi lands\n"
    "    # EARLIER. The rows below therefore quote TOUCHDOWN TIMES t, not ascending phi:\n"
    "    #   trot  phase=0.5 offset=0   bound=0   -> phi FL,RR=.5 ; FR,RL=0 -> td FR,RL@0.00 ; FL,RR@0.50\n"
    "    #   walk  phase=0   offset=.75 bound=.5  -> phi RR=0, FL=.25, RL=.5, FR=.75\n"
    "    #                                        -> td RR@0.00, FR@0.25, RL@0.50, FL@0.75  (4-beat LATERAL)\n"
    "    #      (the old offset=.25 gave phi RR=0, FR=.25, RL=.5, FL=.75 -> td RR@0.00, FL@0.25, RL@0.50,\n"
    "    #       FR@0.75 = the TIME-REVERSED diagonal/primate sequence, not the documented lateral walk.)\n"
)
s = s.replace(old, new)

# ---- (2) walk dict entry: crawl duty + lateral touchdown order ----
old = '        "walk": dict(phase=0.0, offset=0.25, bound=0.5, duration=0.65, pattern="4-beat lateral seq RR->FR->RL->FL"),\n'
assert s.count(old) == 1, "c2"
new = (
    "        # duty 0.78: with the four phis spaced 0.25 the support count is 4*duty, so the old 0.65 gave\n"
    "        # 2.6 feet -- 3 feet only 60% of the cycle and 20% of it on a LATERAL pair (pace-like amble,\n"
    "        # NOT a crawl). 0.75 is exactly 3 feet 100% of the time; 0.78 keeps 3 with margin (12% at 4).\n"
    '        "walk": dict(phase=0.0, offset=0.75, bound=0.5, duration=0.78, pattern="4-beat lateral crawl RR->FR->RL->FL"),\n'
)
s = s.replace(old, new)

# ---- (3) GO2_GAIT_DUTY knob, parsed next to GO2_GAIT ----
old = (
    '        _gait = os.environ.get("GO2_GAIT", "").strip().lower()\n'
    "        if _gait in self._GO2_FIXED_GAITS:\n"
    "            cfg.fixed_gait = _gait\n"
)
assert s.count(old) == 1, "c3"
new = old + (
    "        # GO2_GAIT_DUTY=<float in (0,1)> : override the fixed gait DUTY FACTOR (stance fraction,\n"
    "        # cmd[8]) without editing the gait table. Only meaningful together with GO2_GAIT. The duty\n"
    "        # warp is a pure time reparametrisation, so the real stance fraction == this value, and with\n"
    "        # the walk phis spaced 0.25 the support count is 4*duty (0.75 => a true 3-foot crawl).\n"
    "        # Unset => the gait table duration is used, byte-for-byte as before.\n"
    '        _gduty = os.environ.get("GO2_GAIT_DUTY", "").strip()\n'
    "        if _gduty != \"\" and _gait in self._GO2_FIXED_GAITS:\n"
    "            cfg.fixed_gait_duty = float(_gduty)\n"
)
s = s.replace(old, new)

# ---- (4) store it + use it in the fixed-gait override ----
old = '        self._fixed_gait = getattr(self.cfg, "fixed_gait", None)\n'
assert s.count(old) == 1, "c4"
s = s.replace(old, old + '        self._fixed_gait_duty = getattr(self.cfg, "fixed_gait_duty", None)  # GO2_GAIT_DUTY (wins over the table)\n')

old = (
    "            self._commands[env_ids, 7] = g[\"bound\"]\n"
    "            if self.num_commands > 8:\n"
    "                self._commands[env_ids, 8] = g[\"duration\"]\n"
)
assert s.count(old) == 1, "c5"
new = (
    "            self._commands[env_ids, 7] = g[\"bound\"]\n"
    "            if self.num_commands > 8:\n"
    "                # GO2_GAIT_DUTY wins over the gait-table duration when set.\n"
    "                _duty = getattr(self, \"_fixed_gait_duty\", None)\n"
    "                self._commands[env_ids, 8] = g[\"duration\"] if _duty is None else _duty\n"
)
s = s.replace(old, new)

# ---- (5) banner: predicted support from the ACTUAL schedule ----
old = (
    "        if self._fixed_gait is not None:\n"
    "            g = self._GO2_FIXED_GAITS[self._fixed_gait]\n"
    "            print(\n"
    "                f\"[Go2WTW][gait] FIXED gait='{self._fixed_gait}' pinned: \"\n"
    "                f\"phase(cmd5)={g['phase']} offset(cmd6)={g['offset']} bound(cmd7)={g['bound']} \"\n"
    "                f\"duration(cmd8)={g['duration']} | 4-foot pattern: {g['pattern']}\"\n"
    "            )\n"
)
assert s.count(old) == 1, "c6"
new = (
    "        if self._fixed_gait is not None:\n"
    "            g = self._GO2_FIXED_GAITS[self._fixed_gait]\n"
    "            _duty = self._fixed_gait_duty if self._fixed_gait_duty is not None else g[\"duration\"]\n"
    "            # [SUPPORT-PREDICT] reproduce _contact_target_step exactly (duty warp -> stance is the\n"
    "            # warped half-cycle [0,0.5), kappa=0.07 Gaussian smoothing, threshold 0.5 as the gait\n"
    "            # rewards read it) over one full cycle, so a wrong duty is visible AT LAUNCH.\n"
    "            _th = np.linspace(0.0, 1.0, 2001, endpoint=False)\n"
    "            _phis = [g[\"phase\"] + g[\"offset\"] + g[\"bound\"], g[\"offset\"], g[\"bound\"], g[\"phase\"]]\n"
    "            _n = np.zeros_like(_th)\n"
    "            for _phi in _phis:\n"
    "                _u = np.remainder(_th + _phi, 1.0)\n"
    "                _w = np.where(_u < _duty, _u * (0.5 / _duty), 0.5 + (_u - _duty) * (0.5 / (1.0 - _duty)))\n"
    "                _n += (_w < 0.5).astype(float)  # smoothed value crosses 0.5 exactly at the warped edges\n"
    "            _src = \"GO2_GAIT_DUTY\" if self._fixed_gait_duty is not None else \"table\"\n"
    "            print(\n"
    "                f\"[Go2WTW][gait] FIXED gait='{self._fixed_gait}' pinned: \"\n"
    "                f\"phase(cmd5)={g['phase']} offset(cmd6)={g['offset']} bound(cmd7)={g['bound']} \"\n"
    "                f\"duration(cmd8)={_duty} [{_src}] | 4-foot pattern: {g['pattern']}\"\n"
    "            )\n"
    "            print(\n"
    "                f\"[Go2WTW][gait] predicted support: mean={4.0 * _duty:.3f} feet (4*duty) \"\n"
    "                f\"min={int(_n.min())} max={int(_n.max())} | >=3 feet {100.0 * (_n >= 3).mean():.1f}% \"\n"
    "                f\"of the cycle | measured-schedule mean={_n.mean():.3f}\"\n"
    "            )\n"
)
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8").write(s)
print("PATCH OK")
