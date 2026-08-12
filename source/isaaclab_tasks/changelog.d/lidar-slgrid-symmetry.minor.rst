Added
^^^^^

* Added ``Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0`` with
  ``Go2ParkourImitationLidarSLGridCrawlSymPPOAMPRunnerCfg``, the SL-Grid + crawl arm with the
  L/R mirror data-augmentation restored.  The env cfg is identical to the symmetry-OFF arm, so
  the pair isolates the augmentation — the one recipe difference against the privileged voxel
  teacher, which trained with the mirror ON.
* Added ``obs["lidar"]`` support to the parkour L/R mirror
  (:func:`~isaaclab_tasks.direct.parkour.mdp.symmetry.compute_parkour_symmetric_states`).  The
  SL-Grid metric grid shares the flat (27, 21, 13) C-order voxel layout, so the existing y-flip
  permutation applies; the angular range-image representation is rejected with an explicit
  error instead of being silently passed through unmirrored.
