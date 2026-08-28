Added
^^^^^

* Added the ``Go2-Imitation-Latent-v0`` task, which replaces the AMP discriminator of
  ``Go2-Imitation-Tracking-v0`` with a style reward computed from a frozen motion-VAE
  encoder. The reward is the per-environment time-window marginal KL between the policy's
  latent distribution and the expert reference, so it stays a per-environment signal
  instead of a batch constant.
