Added
^^^^^

* Added ``amp_joint_tan_norm`` to
  :class:`~isaaclab_tasks.direct.go2_imitation_tracking.Go2ImitationTrackingEnvCfg`, which applies
  the tangent/normal joint encoding to the AMP discriminator observation instead of the policy
  observation. Enabling it widens the per-step discriminator observation from 49 to 109, and the
  environment recomputes ``amp_observation_space`` accordingly. Defaults to ``False``.
