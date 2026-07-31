Added
^^^^^

* Added ``Go2-Pedipulation-v0``, a direct RL environment where a standing Unitree Go2 moves a
  commanded foot to a target position in the base frame and holds it there. The command carries a
  per-leg role vector so the manipulating leg is selected by the command rather than hard-coded,
  and the action is routed into stance and manipulation branches so unused channels receive no
  gradient.
* Added a joint-target rate penalty to ``Go2-Pedipulation-v0``, weighted by
  ``w_joint_target_rate``. The existing action-rate penalty acts on the raw action, which is a
  position for stance legs but an increment for the manipulating leg, so the manipulating leg had
  no first-order penalty and a steadily drifting target cost nothing. The new term penalizes the
  commanded joint target's step-to-step change, which is identical for both branches.
* Added ``jvel_filter_alpha`` to ``Go2-Pedipulation-v0``, an exponential moving average on the
  joint-velocity observation. It defaults to ``1.0``, which leaves the observation unfiltered.
  A policy trained with a different value must be evaluated and deployed with the same value.
* Added ``action_filter_alpha`` to ``Go2-Pedipulation-v0``, an exponential moving average on the
  commanded joint target. It defaults to ``1.0``, which leaves the target unfiltered. The
  integrator that feeds the manipulating leg accumulates the pre-filter target, so the plant stays
  an integrator followed by a filter rather than becoming a leaky integrator.
* Added ``CommandCfg.static_fraction`` to ``Go2-Pedipulation-v0``, the share of environments whose
  circle command degenerates to a stationary target. Training only on moving targets makes holding
  a stationary one out of distribution.
