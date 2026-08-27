Fixed
^^^^^

* Fixed :class:`~isaaclab_tasks.direct.hind_leg.HindLegEnv` never terminating an episode when the
  robot falls backward. Termination only checked contact on the ``base`` body, but the legs prop the
  torso up in a backward fall so the base never touches the ground. Policies exploited this and spent
  most of each episode lying down. Added ``terminate_tilt_deg`` (default 60 degrees) and
  ``terminate_base_height`` (default disabled) to :class:`~isaaclab_tasks.direct.hind_leg.HindLegFlatEnvCfg`
  and :class:`~isaaclab_tasks.direct.hind_leg.HindLegHistoryEnvCfg`. Set ``terminate_tilt_deg`` to
  ``None`` to restore the previous contact-only behaviour.
