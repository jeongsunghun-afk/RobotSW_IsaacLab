Added
^^^^^

* Added a ``"command_uniform_mirror"`` option to
  :attr:`~isaaclab_tasks.direct.leg_imitation_tracking.LegImitationTrackingEnvCfg.motion_weight_mode`.
  It synthesizes the left/right mirror of every reference clip that has no mirror partner before
  assigning the command-uniform cell weights, so paired clips always receive equal weight and the
  expert distribution carries no left/right bias.
* Added a ``mirror_complete`` argument and a ``motion_names`` property to
  :class:`~isaaclab_tasks.direct.leg_imitation_tracking.LegMotionLib`.
