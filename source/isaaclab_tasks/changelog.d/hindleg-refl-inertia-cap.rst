Fixed
^^^^^

* Fixed the reflected-inertia coupling term in
  :class:`~isaaclab_tasks.direct.hind_leg.HindLegEnv` and
  :class:`~isaaclab_tasks.direct.r2s_biped_leg.R2SBipedLegEnv` growing without bound. The term is
  an explicit, one-step-delayed feedforward, so it acts as damping of either sign rather than as a
  conservative inertia coupling, and it reached 313 N.m against a 126 N.m joint effort limit during
  locomotion. Added ``foot_reflected_inertia_cap`` (default 25 N.m), a constant clamp chosen above
  the 10.6 N.m maximum the term reaches across the captures the plant was identified on, so the
  identified plant is unchanged. Set it to ``None`` to restore the previous unbounded behaviour.
