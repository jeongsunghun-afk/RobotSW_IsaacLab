Added
^^^^^

* Added left/right symmetry support to the Go2 imitation tracking environment via
  ``mdp.symmetry.compute_go2_symmetric_states``, usable as either data augmentation or a
  mirror-consistency loss. Disabled by default; enable per run with the ``GO2_SYMMETRY_AUG``
  and ``GO2_MIRROR_LOSS`` environment variables.
