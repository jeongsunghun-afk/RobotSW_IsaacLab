Fixed
^^^^^

* Fixed the HindLeg history environment overriding the actuator gains with channel-space
  values. The override predated the gear-ratio conversion in ``HIND_LEG_CFG`` and was never
  updated, so the training environment ran ``calf 50`` while the real2sim environment ran
  ``calf 112.5`` for the same robot. Channel and joint gains are related by
  ``kp_joint = k**2 * kp_ch``, so identical numbers in the two frames are not a match.
  The override also placed the plant-randomisation range off the real value: sampling
  spanned ``[25.0, 75.0]`` at the calf, which never reaches the ``112.5`` the robot presents.
