#!/usr/bin/env python3
"""Anchored patcher: extend GO2_FH_ERR_DUMP with foot labels + add GO2_FOOT_DIAG per-step dump.
Idempotent: refuses to re-apply if the marker is already present."""
import io, sys

P = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
src = io.open(P, encoding="utf-8").read()

if "[FOOT-DIAG]" in src:
    print("ALREADY_PATCHED"); sys.exit(0)

# ---------------- patch 1a: FH_ERR_DUMP, z3d branch ----------------
A1 = '''                        with open(os.environ.get("GO2_FH_ERR_FILE", "/tmp/fh_err.txt"), "a") as _fh_f:
                            for _fh_v, _fh_z in zip(_fh_vals, _fh_vz):
                                _fh_f.write("%.6f %.6f\\n" % (float(_fh_v), float(_fh_z)))
'''
B1 = '''                        # [FOOT-LABEL] columns 1-2 unchanged (xy-err, signed z-err) so every existing
                        # parser keeps working; columns 3+ added for the per-foot swing diagnosis:
                        #   3   foot index (0=FL 1=FR 2=RL 3=RR)
                        #   4-6 foot xyz (world)   7-9 foothold target xyz (world)
                        #   10  desired_contact_states at this instant   11 gait phase foot_indices
                        #   12  ep_step
                        _fh_ij = _fh_m.nonzero(as_tuple=False).detach().cpu().numpy()
                        _fh_fp = foot_positions[_fh_m].detach().cpu().numpy()
                        _fh_tg = self._foothold_target[_fh_m].detach().cpu().numpy()
                        _fh_dc = self.desired_contact_states[_fh_m].detach().cpu().numpy()
                        _fh_ph = self.foot_indices[_fh_m].detach().cpu().numpy()
                        _fh_eb = self.episode_length_buf.detach().cpu().numpy()
                        with open(os.environ.get("GO2_FH_ERR_FILE", "/tmp/fh_err.txt"), "a") as _fh_f:
                            for _fh_k in range(len(_fh_vals)):
                                _fh_f.write(
                                    "%.6f %.6f %d %.5f %.5f %.5f %.5f %.5f %.5f %.4f %.4f %d\\n"
                                    % (
                                        float(_fh_vals[_fh_k]),
                                        float(_fh_vz[_fh_k]),
                                        int(_fh_ij[_fh_k, 1]),
                                        float(_fh_fp[_fh_k, 0]), float(_fh_fp[_fh_k, 1]), float(_fh_fp[_fh_k, 2]),
                                        float(_fh_tg[_fh_k, 0]), float(_fh_tg[_fh_k, 1]), float(_fh_tg[_fh_k, 2]),
                                        float(_fh_dc[_fh_k]),
                                        float(_fh_ph[_fh_k]),
                                        int(_fh_eb[int(_fh_ij[_fh_k, 0])]),
                                    )
                                )
'''
assert src.count(A1) == 1, "anchor A1 not unique: %d" % src.count(A1)
src = src.replace(A1, B1)

# ---------------- patch 1b: FH_ERR_DUMP, xy-only branch ----------------
A2 = '''                        with open(os.environ.get("GO2_FH_ERR_FILE", "/tmp/fh_err.txt"), "a") as _fh_f:
                            for _fh_v in _fh_vals:
                                _fh_f.write("%.6f\\n" % float(_fh_v))
'''
B2 = '''                        # [FOOT-LABEL] column 1 unchanged (xy-err); columns 2+ = foot index, foot xyz,
                        # target xyz, desired_contact_states, gait phase, ep_step.
                        _fh_ij = _fh_m.nonzero(as_tuple=False).detach().cpu().numpy()
                        _fh_fp = foot_positions[_fh_m].detach().cpu().numpy()
                        _fh_tg = self._foothold_target[_fh_m].detach().cpu().numpy()
                        _fh_dc = self.desired_contact_states[_fh_m].detach().cpu().numpy()
                        _fh_ph = self.foot_indices[_fh_m].detach().cpu().numpy()
                        _fh_eb = self.episode_length_buf.detach().cpu().numpy()
                        with open(os.environ.get("GO2_FH_ERR_FILE", "/tmp/fh_err.txt"), "a") as _fh_f:
                            for _fh_k in range(len(_fh_vals)):
                                _fh_f.write(
                                    "%.6f %d %.5f %.5f %.5f %.5f %.5f %.5f %.4f %.4f %d\\n"
                                    % (
                                        float(_fh_vals[_fh_k]),
                                        int(_fh_ij[_fh_k, 1]),
                                        float(_fh_fp[_fh_k, 0]), float(_fh_fp[_fh_k, 1]), float(_fh_fp[_fh_k, 2]),
                                        float(_fh_tg[_fh_k, 0]), float(_fh_tg[_fh_k, 1]), float(_fh_tg[_fh_k, 2]),
                                        float(_fh_dc[_fh_k]),
                                        float(_fh_ph[_fh_k]),
                                        int(_fh_eb[int(_fh_ij[_fh_k, 0])]),
                                    )
                                )
'''
assert src.count(A2) == 1, "anchor A2 not unique: %d" % src.count(A2)
src = src.replace(A2, B2)

# ---------------- patch 2: GO2_FOOT_DIAG per-step dump ----------------
A3 = '''                    + " ".join("%.4f" % float(_v) for _v in _cd_dv)
                    + "\\n"
                )
'''
B3 = A3 + '''
        # [FOOT-DIAG] GO2_FOOT_DIAG=1 -> env-0 per-step per-foot record for the "RL never swings"
        # diagnosis (A kinematic reach / B support-polygon stability / C reward gate).
        # ONE line per control step, whitespace separated:
        #   col  1        ep_step
        #   col  2- 4     base xyz (world)
        #   col  5        base yaw (rad)
        #   col  6-69     16 values per foot, feet in order FL,FR,RL,RR:
        #                   fx fy fz    foot link position, world
        #                   tx ty tz    foothold target (self._foothold_target), world
        #                   thx thy thz (target - hip) expressed in the BASE frame  <- reachability
        #                   fhx fhy fhz (foot   - hip) expressed in the BASE frame  <- achieved workspace
        #                   c           measured contact (max-history |F| > GO2_CONTACT_THRESH) 0/1
        #                   fc          contact-sensor first_contact this step 0/1
        #                   d           desired_contact_states in [0,1]  (commanded stance)
        #                   s           gait phase self.foot_indices in [0,1)
        #   col 70        n_meas   (# feet in measured contact)
        #   col 71        margin_act   signed distance base-xy -> boundary of the CURRENT support
        #                              polygon (feet in measured contact), metres, + = inside
        #   col 72-75     margin_wo_FL .. margin_wo_RR : the same signed margin for the polygon of the
        #                 OTHER THREE feet (all four foot xy, foot i removed) = the static margin the
        #                 robot would have if foot i lifted right now.  Decisive for hypothesis (B).
        # Off by default; byte-identical behaviour when unset.
        if os.environ.get("GO2_FOOT_DIAG", "") == "1":

            def _fd_margin(_pts, _p):
                # signed distance from _p (2,) to the boundary of the convex polygon of _pts (k,2).
                # + = inside.  k>=3: min over CCW edges of the inward signed line distance (exact
                # inside, conservative outside).  k==2: -distance to the segment. k<=1: degenerate.
                _k = int(_pts.shape[0])
                if _k == 0:
                    return -9.0
                if _k == 1:
                    return -float(np.linalg.norm(_pts[0] - _p))
                if _k == 2:
                    _a = _pts[0]
                    _ab = _pts[1] - _a
                    _l2 = float(_ab.dot(_ab))
                    _t = 0.0 if _l2 <= 1e-12 else max(0.0, min(1.0, float((_p - _a).dot(_ab)) / _l2))
                    return -float(np.linalg.norm(_a + _t * _ab - _p))
                _c = _pts.mean(axis=0)
                _o = _pts[np.argsort(np.arctan2(_pts[:, 1] - _c[1], _pts[:, 0] - _c[0]))]  # CCW
                _m = 1e9
                for _i in range(_k):
                    _a = _o[_i]
                    _e = _o[(_i + 1) % _k] - _a
                    _n = float(np.linalg.norm(_e))
                    if _n < 1e-9:
                        continue
                    _sd = float((-_e[1] * (_p[0] - _a[0]) + _e[0] * (_p[1] - _a[1])) / _n)
                    if _sd < _m:
                        _m = _sd
                return float(_m)

            _fd_thr = float(os.environ.get("GO2_CONTACT_THRESH", "1.0"))
            _fd_frc = torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_ids], dim=-1), dim=1)[0][0]
            _fd_qi = inv_quat(base_quat).unsqueeze(1).expand(-1, 4, -1).reshape(self.num_envs * 4, 4)
            _fd_hipw = self._robot.data.body_link_pos_w[:, self._hip_body_ids, :]
            _fd_hb = quat_apply(_fd_qi, (_fd_hipw - base_pos.unsqueeze(1)).reshape(self.num_envs * 4, 3)).view(
                self.num_envs, 4, 3
            )
            _fd_tb = quat_apply(
                _fd_qi, (self._foothold_target - base_pos.unsqueeze(1)).reshape(self.num_envs * 4, 3)
            ).view(self.num_envs, 4, 3)
            _fd_fb = quat_apply(_fd_qi, (foot_positions - base_pos.unsqueeze(1)).reshape(self.num_envs * 4, 3)).view(
                self.num_envs, 4, 3
            )
            _fd_fp = foot_positions[0].detach().cpu().numpy()
            _fd_tg = self._foothold_target[0].detach().cpu().numpy()
            _fd_th = (_fd_tb - _fd_hb)[0].detach().cpu().numpy()
            _fd_fh = (_fd_fb - _fd_hb)[0].detach().cpu().numpy()
            _fd_bp = base_pos[0].detach().cpu().numpy()
            _fd_q = base_quat[0].detach().cpu().numpy()
            _fd_yaw = float(
                np.arctan2(
                    2.0 * (_fd_q[0] * _fd_q[3] + _fd_q[1] * _fd_q[2]),
                    1.0 - 2.0 * (_fd_q[2] * _fd_q[2] + _fd_q[3] * _fd_q[3]),
                )
            )
            _fd_c = (_fd_frc > _fd_thr).detach().cpu().numpy()
            _fd_fc = first_contact[0].detach().cpu().numpy()
            _fd_d = self.desired_contact_states[0].detach().cpu().numpy()
            _fd_s = self.foot_indices[0].detach().cpu().numpy()
            _fd_p = _fd_bp[:2]
            _fd_xy = _fd_fp[:, :2]
            _fd_act = _fd_margin(_fd_xy[_fd_c.astype(bool)], _fd_p)
            _fd_wo = [_fd_margin(_fd_xy[[_j for _j in range(4) if _j != _i]], _fd_p) for _i in range(4)]
            _fd_cols = ["%d" % int(self.episode_length_buf[0].item())]
            _fd_cols += ["%.5f" % float(_v) for _v in _fd_bp]
            _fd_cols += ["%.5f" % _fd_yaw]
            for _i in range(4):
                _fd_cols += ["%.5f" % float(_v) for _v in _fd_fp[_i]]
                _fd_cols += ["%.5f" % float(_v) for _v in _fd_tg[_i]]
                _fd_cols += ["%.5f" % float(_v) for _v in _fd_th[_i]]
                _fd_cols += ["%.5f" % float(_v) for _v in _fd_fh[_i]]
                _fd_cols += ["%d" % int(bool(_fd_c[_i]))]
                _fd_cols += ["%d" % int(bool(_fd_fc[_i]))]
                _fd_cols += ["%.4f" % float(_fd_d[_i])]
                _fd_cols += ["%.4f" % float(_fd_s[_i])]
            _fd_cols += ["%d" % int(_fd_c.sum())]
            _fd_cols += ["%.5f" % _fd_act]
            _fd_cols += ["%.5f" % _v for _v in _fd_wo]
            with open(os.environ.get("GO2_FOOT_DIAG_FILE", "/tmp/foot_diag.txt"), "a") as _fd_fh_f:
                _fd_fh_f.write(" ".join(_fd_cols) + "\\n")
'''
assert src.count(A3) == 1, "anchor A3 not unique: %d" % src.count(A3)
src = src.replace(A3, B3)

io.open(P, "w", encoding="utf-8").write(src)
print("PATCHED_OK")
