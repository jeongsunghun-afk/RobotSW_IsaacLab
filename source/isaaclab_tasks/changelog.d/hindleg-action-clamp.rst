Fixed
^^^^^

* Fixed the HindLeg environment sending joint position targets outside the reachable range.
  The real bridge clamps targets to the soft joint limits, but the environment did not, so
  the policy learned to hold targets far beyond the limit and use the resulting standing
  position error as a torque saturator. At ``cmd 1.0`` this put 81.6 percent of foot targets
  and 32.0 percent of hip targets outside their limits, and every motor-saturating step
  coincided with an out-of-limit target. Targets are now clamped to
  :attr:`~isaaclab.assets.ArticulationData.soft_joint_pos_limits`, matching the bridge.
