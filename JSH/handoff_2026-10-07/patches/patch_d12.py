import io, sys

F = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
src = io.open(F, encoding="utf-8").read()
orig = src

# ---------------------------------------------------------------- 1) module-level shape function
A1 = '''def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower
'''
N1 = '''def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def go2_swing_force_p(F, mode, fref=25.0, tail=0.05, pmax=1.2, sigma=3000.0, k=1.0):
    """[D12] Swing-force penalty shape p(F) for tracking_contacts_shaped_force = -(1-d)*p(F).

    The pre-D12 shape is p = 1 - exp(-F^2/sigma) with sigma = 100. It is a near-STEP: p(24.6 N) =
    0.9976 with dp/dF = 1.2e-3 /N, and 1.0000 with dp/dF = 1.7e-8 /N at 42.1 N. 24.6 N is the
    measured median normal force on the stuck RL foot during RL's OWN commanded swing and 42.1 N is
    the load at which the swing is missed -- exactly the band in which the term must pay the policy
    to unload, and exactly where it is flat.

    D11 widened it (sigma 3000). That restored the gradient (1.34e-2 /N) but dropped the LEVEL to
    p(24.6 N) = 0.1827, making a planted swing foot 5.5x cheaper; the policy stopped lifting (support
    3.73 feet, 0/11 rows). The Huber variant shares the defect (p(24.6 N) = 0.135).

    Both modes here keep the LEVEL at the operating point and add gradient:
      "linear": p = F/fref for F <= fref, then 1 + tail*(F/fref - 1), clamped to [0, pmax].
                fref = 25 N => p(24.6) = 0.984 (level kept) and dp/dF = 1/fref = 4.0e-2 /N CONSTANT
                over the whole 0..fref band the unloading foot traverses; the tail keeps dp/dF =
                tail/fref > 0 above fref so the term is never exactly flat.
      "scaled": p = k*(1 - exp(-F^2/sigma)) with k = p(24.6; 100)/p(24.6; sigma) (~5.46 at sigma =
                3000), clamped to [0, pmax]. Level-preserving AT the operating point by construction,
                but since the original p there is already 0.9976 the clamp binds at ~24.6 N, so this
                mode is FLAT at and above the operating point (and, unclamped, overshoots to 5.3x the
                original level at 100 N). Provided for the ablation; see the [D12] banner.

    p(0) = 0 in every mode, so the three healthy feet (median 0.0 N during their own swings) are
    untouched. Autograd-safe: both torch.where branches are finite everywhere.
    """
    if mode == "linear":
        r = F / fref
        return torch.clamp(torch.where(r <= 1.0, r, 1.0 + tail * (r - 1.0)), min=0.0, max=pmax)
    if mode == "scaled":
        return torch.clamp(k * (1.0 - torch.exp(-(F * F) / sigma)), min=0.0, max=pmax)
    raise ValueError("GO2_GAIT_FORCE_MODE must be 'linear' or 'scaled', got %r" % (mode,))
'''
assert src.count(A1) == 1, "anchor 1"
src = src.replace(A1, N1)

# ---------------------------------------------------------------- 2) parse + banner in __init__
A2 = '''        if self._fh_gate_soft:
            print(
                "[Go2WTW][D11] FH-GATE-SOFT ON (GO2_FH_GATE_SOFT=1): foothold_track gate = "'''
N2 = '''        # [D12] GO2_GAIT_FORCE_MODE=<linear|scaled> -- LEVEL-PRESERVING swing-force penalty.
        #   D11's sigma 100 -> 3000 fixed the SHAPE but regressed the LEVEL: p(24.6 N) 0.9976 ->
        #   0.1827, i.e. holding a foot planted through its own commanded swing became 5.5x CHEAPER.
        #   The policy took that deal exactly as offered -- measured support rose to 3.728 feet with
        #   four-foot support 75.8% of the time (commanded 3.12), swings completed over the 70-cycle
        #   frozen phase were FL 2/70, RL 1/70, RR 0/70, FR 70/70, it froze at the field edge
        #   (base_x max 0.792 m vs row 0's leading edge 0.822 m), 0/11 rows, reproducible on a second
        #   seed. GO2_GAIT_FORCE_SIGMA=huber has the same defect (p(24.6 N) = 0.135).
        #   These modes keep the level AND add gradient; see go2_swing_force_p() for the shapes.
        #   Acceptance at the F = 24.6 N operating point: p >= 0.95 AND dp/dF >= 5e-3 /N.
        #   Knobs (all optional): GO2_GAIT_FORCE_FREF (linear, default 25 N), GO2_GAIT_FORCE_TAIL
        #   (linear tail slope ratio, default 0.05), GO2_GAIT_FORCE_PMAX (default 1.2 linear / 1.0
        #   scaled), GO2_GAIT_FORCE_SIGMA (scaled only, default 3000).
        #   Unset => none of this runs and the penalty is byte-identical to the pre-D12 code.
        _d12_m = _os_std.environ.get("GO2_GAIT_FORCE_MODE", "").strip().lower()
        if _d12_m not in ("", "linear", "scaled"):
            raise ValueError(f"GO2_GAIT_FORCE_MODE={_d12_m!r} unknown (expected 'linear' or 'scaled')")
        self._gait_force_mode = _d12_m
        if _d12_m != "":
            if self._gait_force_huber:
                print("[Go2WTW][D12][warn] GO2_GAIT_FORCE_MODE overrides GO2_GAIT_FORCE_SIGMA=huber")
                self._gait_force_huber = False
            self._gait_force_fref = float(_os_std.environ.get("GO2_GAIT_FORCE_FREF", "25.0"))
            self._gait_force_tail = float(_os_std.environ.get("GO2_GAIT_FORCE_TAIL", "0.05"))
            self._gait_force_pmax = float(
                _os_std.environ.get("GO2_GAIT_FORCE_PMAX", "1.0" if _d12_m == "scaled" else "1.2")
            )
            try:
                self._gait_force_sigma_new = float(_os_std.environ.get("GO2_GAIT_FORCE_SIGMA", "3000"))
            except ValueError:  # e.g. the literal "huber"
                self._gait_force_sigma_new = 3000.0
            _p100 = float(1.0 - np.exp(-(_d11_fop**2) / 100.0))
            self._gait_force_k = _p100 / float(1.0 - np.exp(-(_d11_fop**2) / self._gait_force_sigma_new))

            def _d12_pg(_f, _mode):  # -> (p, dp/dF) analytic, numpy, for the banner table
                if _mode == "sigma100":
                    return (
                        float(1.0 - np.exp(-(_f**2) / 100.0)),
                        float((2.0 * _f / 100.0) * np.exp(-(_f**2) / 100.0)),
                    )
                if _mode == "sigma3000":
                    return (
                        float(1.0 - np.exp(-(_f**2) / 3000.0)),
                        float((2.0 * _f / 3000.0) * np.exp(-(_f**2) / 3000.0)),
                    )
                if _mode == "linear":
                    _fr, _t, _pm = self._gait_force_fref, self._gait_force_tail, self._gait_force_pmax
                    _r = _f / _fr
                    _p = _r if _r <= 1.0 else 1.0 + _t * (_r - 1.0)
                    _g = (1.0 / _fr) if _r <= 1.0 else (_t / _fr)
                    return (min(_p, _pm), 0.0 if _p >= _pm else _g)
                _s, _k, _pm = self._gait_force_sigma_new, self._gait_force_k, self._gait_force_pmax
                _p = _k * (1.0 - np.exp(-(_f**2) / _s))
                _g = _k * (2.0 * _f / _s) * np.exp(-(_f**2) / _s)
                return (float(min(_p, _pm)), 0.0 if _p >= _pm else float(_g))

            _pn, _gn = _d12_pg(_d11_fop, _d12_m)
            _po, _go = _d12_pg(_d11_fop, "sigma100")
            _ok = "PASS" if (_pn >= 0.95 and _gn >= 5e-3) else "FAIL"
            _par = (
                f"F_ref={self._gait_force_fref:g} N tail={self._gait_force_tail:g} pmax={self._gait_force_pmax:g}"
                if _d12_m == "linear"
                else f"k={self._gait_force_k:.4f} sigma={self._gait_force_sigma_new:g} pmax={self._gait_force_pmax:g}"
            )
            print(
                f"[Go2WTW][D12] GAIT-FORCE-MODE ON (GO2_GAIT_FORCE_MODE={_d12_m}): swing-force penalty "
                f"-(1-d)*p(F) with the LEVEL-PRESERVING p, {_par}. At the measured operating point "
                f"F={_d11_fop} N: p {_po:.4f} -> {_pn:.4f} (level kept) and dp/dF {_go:.3e} -> {_gn:.3e} /N "
                f"({_gn / _go:.1f}x the gradient). ACCEPTANCE (p>=0.95 and dp/dF>=5e-3 /N): {_ok}"
            )
            print("[Go2WTW][D12]   F [N] |    p sig100  dp/dF |    p sig3000  dp/dF |    p %-7s  dp/dF" % _d12_m)
            for _f in (0.0, 5.0, 10.0, 24.6, 42.1, 100.0):
                _a = _d12_pg(_f, "sigma100")
                _b = _d12_pg(_f, "sigma3000")
                _c = _d12_pg(_f, _d12_m)
                print(
                    f"[Go2WTW][D12]   {_f:5.1f} | {_a[0]:9.4f} {_a[1]:9.2e} | {_b[0]:9.4f} {_b[1]:9.2e} |"
                    f" {_c[0]:9.4f} {_c[1]:9.2e}"
                )
            if _ok == "FAIL":
                raise ValueError("[Go2WTW][D12] acceptance criterion FAILED at the operating point")
        if self._fh_gate_soft:
            print(
                "[Go2WTW][D11] FH-GATE-SOFT ON (GO2_FH_GATE_SOFT=1): foothold_track gate = "'''
assert src.count(A2) == 1, "anchor 2"
src = src.replace(A2, N2)

# ---------------------------------------------------------------- 3) reward branch
A3 = '''                tracking_contacts_shaped_force += -((1 - desired_contact[:, i]) * _d11_p)
            tracking_contacts_shaped_force = tracking_contacts_shaped_force / 4
        tracking_contacts_shaped_force[both_low] = 0.0'''
N3 = '''                tracking_contacts_shaped_force += -((1 - desired_contact[:, i]) * _d11_p)
            tracking_contacts_shaped_force = tracking_contacts_shaped_force / 4
        elif getattr(self, "_gait_force_mode", "") != "":
            # [D12] LEVEL-PRESERVING swing-force penalty -- recompute -(1-d)*p(F)/4 with the p from
            # go2_swing_force_p(): same operating-point LEVEL as the sigma=100 original (p(24.6 N)
            # >= 0.95, so a planted swing foot costs what it used to) but with a live gradient there
            # (dp/dF >= 5e-3 /N) so the policy is actually paid to unload. p(0) = 0 as before, so the
            # feet that already swing clean (median 0.0 N) see no change at all.
            _d12_p = go2_swing_force_p(
                foot_forces,
                self._gait_force_mode,
                fref=self._gait_force_fref,
                tail=self._gait_force_tail,
                pmax=self._gait_force_pmax,
                sigma=self._gait_force_sigma_new,
                k=self._gait_force_k,
            )
            tracking_contacts_shaped_force = -torch.sum((1.0 - desired_contact) * _d12_p, dim=1) / 4
        tracking_contacts_shaped_force[both_low] = 0.0'''
assert src.count(A3) == 1, "anchor 3"
src = src.replace(A3, N3)

assert src != orig
io.open(F, "w", encoding="utf-8").write(src)
print("PATCH OK  bytes %d -> %d" % (len(orig), len(src)))
