Added
^^^^^

* Added ``Go2-Pedipulation-v0``, a direct RL environment where a standing Unitree Go2 moves a
  commanded foot to a target position in the base frame and holds it there. The command carries a
  per-leg role vector so the manipulating leg is selected by the command rather than hard-coded,
  and the action is routed into stance and manipulation branches so unused channels receive no
  gradient.
