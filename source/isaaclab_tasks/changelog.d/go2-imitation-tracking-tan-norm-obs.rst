Added
^^^^^

* Added ``joint_pos_tan_norm`` to
  :class:`~isaaclab_tasks.direct.go2_imitation_tracking.Go2ImitationTrackingEnvCfg`, which encodes
  each joint angle as the tangent/normal pair of its rotation (6 values per joint) instead of a raw
  angle. Enabling it widens the policy observation from 42 to 102, and the environment recomputes
  ``observation_space`` accordingly. Defaults to ``False``, so existing checkpoints are unaffected.
