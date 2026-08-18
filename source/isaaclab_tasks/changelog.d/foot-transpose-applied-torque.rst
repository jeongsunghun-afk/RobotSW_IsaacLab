Changed
^^^^^^^

* Changed the foot-to-calf transmission coupling in the ``hind_leg`` and ``r2s_biped_leg`` direct
  environments to load the calf with the foot actuator's realized torque
  (``ArticulationData.applied_torque``) instead of recomputing the foot PD output and clamping it at
  the static ``effort_limit``. The static clamp ignored the DC motor torque-speed curve, so the
  transmitted torque could be over-estimated whenever the foot joint ran fast enough for the curve
  to fall below ``effort_limit`` (beyond about 4.92 rad/s). Because ``applied_torque`` already
  contains the raw-coordinate friction feedforward and is already clipped by the motor model, the
  friction term is no longer added a second time and the result is no longer re-clamped.
  ``R2SBipedLegEnv._apply_foot_coupling`` no longer takes the ``kp_f`` and ``kd_f`` arguments.
