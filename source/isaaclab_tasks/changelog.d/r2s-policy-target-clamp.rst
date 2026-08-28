Fixed
^^^^^

* Fixed the real2sim policy path sending joint position targets outside the reachable range.
  ``set_policy_target`` passed them straight through while the real bridge clamps to the soft
  limits, so the deploy rehearsal ran a plant that matched neither training nor the robot.
  Targets are now clamped to
  :attr:`~isaaclab.assets.ArticulationData.soft_joint_pos_limits`, matching both.
