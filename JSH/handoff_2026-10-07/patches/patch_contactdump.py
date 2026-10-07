import io, shutil, sys

F = "/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py"
BAK = F + ".bak_contactdump"

src = io.open(F, "r", encoding="utf-8").read()
if "GO2_CONTACT_DUMP" in src:
    print("ALREADY_PATCHED"); sys.exit(0)

anchor = '''        if os.environ.get("GO2_RAY_DUMP", "") == "1" and getattr(self, "_scan_enabled", False):
            _rz0 = self._height_scanner.data.ray_hits_w[0, :, 2].detach().cpu().numpy()
            with open(os.environ.get("GO2_RAY_FILE", "/tmp/ray_z.txt"), "a") as _rz_f:
                _rz_f.write(" ".join("%.4f" % float(_v) for _v in _rz0) + "\\n")
'''
assert src.count(anchor) == 1, src.count(anchor)

add = '''
        # [GAIT-DUTY] GO2_CONTACT_DUMP=1 -> append env-0's per-step stance bookkeeping so the
        # "how many feet are actually down at once" question is answerable offline (walk = static
        # crawl 3-support?  vs 2-3 amble). One line per control step, 11 columns:
        #   ep_step  n_meas  n_des  f_FL f_FR f_RL f_RR  d_FL d_FR d_RL d_RR
        # n_meas = # feet whose MEASURED contact force passes the same test _get_dones / the
        # undesired-contact reward use (max over the force history, norm > GO2_CONTACT_THRESH,
        # default 1.0 N); f_* = that per-foot force magnitude in N.  n_des = # feet with
        # desired_contact_states > 0.5 (the COMMANDED schedule, exactly as the gait rewards read
        # it); d_* = the raw (von-Mises-smoothed) desired_contact_states in [0,1].
        # Off by default; byte-identical behaviour when unset.
        if os.environ.get("GO2_CONTACT_DUMP", "") == "1":
            _cd_thr = float(os.environ.get("GO2_CONTACT_THRESH", "1.0"))
            _cd_f = torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_ids], dim=-1), dim=1)[0][0]
            _cd_d = self.desired_contact_states[0]
            _cd_nm = int((_cd_f > _cd_thr).sum().item())
            _cd_nd = int((_cd_d > 0.5).sum().item())
            _cd_fv = _cd_f.detach().cpu().numpy()
            _cd_dv = _cd_d.detach().cpu().numpy()
            with open(os.environ.get("GO2_CONTACT_FILE", "/tmp/contact_dump.txt"), "a") as _cd_fh:
                _cd_fh.write(
                    "%d %d %d " % (int(self.episode_length_buf[0].item()), _cd_nm, _cd_nd)
                    + " ".join("%.3f" % float(_v) for _v in _cd_fv)
                    + " "
                    + " ".join("%.4f" % float(_v) for _v in _cd_dv)
                    + "\\n"
                )
'''

shutil.copy2(F, BAK)
io.open(F, "w", encoding="utf-8").write(src.replace(anchor, anchor + add))
print("PATCHED")
