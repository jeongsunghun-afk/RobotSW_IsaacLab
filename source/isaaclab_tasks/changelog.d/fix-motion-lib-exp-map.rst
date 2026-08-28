Fixed
^^^^^

* Fixed ``motion_lib`` in the Go2 imitation and parkour imitation tasks reading the reference
  motion root rotation as ZYX Euler angles when the motion files store an exponential map
  (axis times angle). The misinterpretation grew with rotation magnitude, reaching 12 degrees
  of attitude error on turning clips.
* Fixed reference motion angular velocity being derived from Euler-angle finite differences,
  which produced spikes of up to 373 rad/s whenever yaw wrapped across the plus/minus pi
  boundary. Angular velocity is now computed from consecutive quaternions, matching the
  retargeting pipeline that generated the motion files.
