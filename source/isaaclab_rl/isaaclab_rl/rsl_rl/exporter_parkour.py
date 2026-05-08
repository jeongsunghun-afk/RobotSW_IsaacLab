# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn


def export_policy_as_jit_parkour(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx_parkour(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


def _infer_dim(module, fallback=None):
    """Infer feature dim from EmpiricalNormalization's _mean buffer shape if available.

    EmpiricalNormalization registers _mean with shape [1, dim], so _mean.shape[-1] gives the feature dim.
    Falls back to `fallback` if the module has no _mean (e.g. nn.Identity).
    """
    if hasattr(module, "_mean"):
        return module._mean.shape[-1]
    if fallback is not None:
        return fallback
    raise ValueError(f"Cannot infer dimension from module {type(module)}: no _mean attribute and no fallback provided.")


class _TorchPolicyExporter(nn.Module):
    """Exporter of actor-critic into JIT file.

    Supports three policy structures dynamically:
    - Simple ActorCritic: single proprio input
    - ActorCriticRMA (multi-encoder): proprio + [history] + [scan]
    - Recurrent ActorCritic: LSTM/GRU hidden state

    Forward order mirrors act_inference: proprio_normed -> history_latent -> scan_latent.
    Scan is NOT normalized (mirrors ActorCriticRMA.act_inference which skips scan_obs_normalizer).
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent

        # --- Actor ---
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        # --- Recurrent setup ---
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        # --- Actor proprio normalizer ---
        if normalizer is not None:
            self.actor_obs_normalizer = copy.deepcopy(normalizer)
        elif hasattr(policy, "actor_obs_normalizer"):
            self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # --- Dynamic encoder detection (non-recurrent multi-encoder path) ---
        # History encoder: present in ActorCriticRMA
        self._has_history = (not self.is_recurrent) and hasattr(policy, "history_encoder") and policy.history_encoder is not None
        if self._has_history:
            self.history_encoder = copy.deepcopy(policy.history_encoder)
            if hasattr(policy, "history_obs_normalizer"):
                self.history_obs_normalizer = copy.deepcopy(policy.history_obs_normalizer)
            else:
                self.history_obs_normalizer = nn.Identity()
            # Dim: history_encoder is StateHistoryEncoder; input is proprio * tsteps (flattened)
            self._history_dim = _infer_dim(
                self.history_obs_normalizer,
                fallback=self.history_encoder.encoder[0].in_features * self.history_encoder.tsteps,
            )

        # Scan encoder: present in ActorCriticRMA when num_scan_obs > 0
        # scandot_encoder is set to None when no scan, so check both attribute existence and non-None
        self._has_scan = (
            (not self.is_recurrent)
            and hasattr(policy, "scandot_encoder")
            and policy.scandot_encoder is not None
        )
        if self._has_scan:
            self.scandot_encoder = copy.deepcopy(policy.scandot_encoder)
            # NOTE: scan_obs_normalizer is NOT applied at inference in ActorCriticRMA.act_inference.
            # The normalizer is only used during training update_normalization. We mirror that behavior
            # here: scan goes directly into scandot_encoder without normalization.
            # Dim: scandot_encoder is an MLP; first Linear's in_features
            self._scan_dim = self.scandot_encoder[0].in_features

        # priv_only fallback: policy has priv_encoder but no history_encoder (unusual but handled)
        self._has_priv_only = (
            (not self.is_recurrent)
            and (not self._has_history)
            and hasattr(policy, "priv_encoder")
            and policy.priv_encoder is not None
        )
        if self._has_priv_only:
            self.priv_encoder = copy.deepcopy(policy.priv_encoder)
            if hasattr(policy, "priv_obs_normalizer"):
                self.priv_obs_normalizer = copy.deepcopy(policy.priv_obs_normalizer)
            else:
                self.priv_obs_normalizer = nn.Identity()
            self._priv_dim = _infer_dim(
                self.priv_obs_normalizer,
                fallback=self.priv_encoder[0].in_features,
            )

        # Infer proprio dim for dummy input construction
        self._proprio_dim = _infer_dim(self.actor_obs_normalizer, fallback=None)
        if self._proprio_dim is None:
            # Identity normalizer with no _mean: try to infer from actor input and encoder output dims
            actor_in = self.actor[0].in_features
            encoder_out = 0
            if self._has_history:
                encoder_out += self.history_encoder.linear_output[-2].out_features  # StateHistoryEncoder output
            if self._has_scan:
                encoder_out += self.scandot_encoder[-1].out_features
            if self._has_priv_only:
                encoder_out += self.priv_encoder[-1].out_features
            self._proprio_dim = actor_in - encoder_out

    def forward_lstm(self, x):
        x = self.actor_obs_normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.actor_obs_normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, proprio, *extras):
        """Dynamic forward mirroring ActorCriticRMA.act_inference order:
        proprio_normed -> history_latent -> scan_latent.

        For simple ActorCritic (no encoders), extras is empty and only proprio is used.
        """
        actor_input_parts = [self.actor_obs_normalizer(proprio)]
        extras_idx = 0

        if self._has_history:
            history = extras[extras_idx]
            extras_idx += 1
            history_latent = self.history_encoder(self.history_obs_normalizer(history))
            actor_input_parts.append(history_latent)
        elif self._has_priv_only:
            priv = extras[extras_idx]
            extras_idx += 1
            priv_latent = self.priv_encoder(self.priv_obs_normalizer(priv))
            actor_input_parts.append(priv_latent)

        if self._has_scan:
            scan = extras[extras_idx]
            # No scan normalizer applied here — mirrors act_inference behavior
            scan_latent = self.scandot_encoder(scan)
            actor_input_parts.append(scan_latent)

        return self.actor(torch.cat(actor_input_parts, dim=-1))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        # Build dummy inputs to trace
        dummy_inputs = [torch.zeros(1, self._proprio_dim)]
        if self._has_history:
            dummy_inputs.append(torch.zeros(1, self._history_dim))
        elif self._has_priv_only:
            dummy_inputs.append(torch.zeros(1, self._priv_dim))
        if self._has_scan:
            dummy_inputs.append(torch.zeros(1, self._scan_dim))
        traced_script_module = torch.jit.trace(self, tuple(dummy_inputs))
        traced_script_module.save(path)


class _OnnxPolicyExporter(nn.Module):
    """Exporter of actor-critic into ONNX file.

    Supports three policy structures dynamically:
    - Simple ActorCritic: single proprio input
    - ActorCriticRMA (multi-encoder): proprio + [history] + [scan]
    - Recurrent ActorCritic: LSTM/GRU hidden state

    Forward order mirrors act_inference: proprio_normed -> history_latent -> scan_latent.
    Scan is NOT normalized (mirrors ActorCriticRMA.act_inference which skips scan_obs_normalizer).
    """

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent

        # --- Actor ---
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        # --- Recurrent setup ---
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        # --- Actor proprio normalizer ---
        if normalizer is not None:
            self.actor_obs_normalizer = copy.deepcopy(normalizer)
        elif hasattr(policy, "actor_obs_normalizer"):
            self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # --- Dynamic encoder detection (non-recurrent multi-encoder path) ---
        # History encoder: present in ActorCriticRMA
        self._has_history = (not self.is_recurrent) and hasattr(policy, "history_encoder") and policy.history_encoder is not None
        if self._has_history:
            self.history_encoder = copy.deepcopy(policy.history_encoder)
            if hasattr(policy, "history_obs_normalizer"):
                self.history_obs_normalizer = copy.deepcopy(policy.history_obs_normalizer)
            else:
                self.history_obs_normalizer = nn.Identity()
            # Dim: history_encoder is StateHistoryEncoder; input is proprio * tsteps (flattened)
            self._history_dim = _infer_dim(
                self.history_obs_normalizer,
                fallback=self.history_encoder.encoder[0].in_features * self.history_encoder.tsteps,
            )

        # Scan encoder: present in ActorCriticRMA when num_scan_obs > 0
        self._has_scan = (
            (not self.is_recurrent)
            and hasattr(policy, "scandot_encoder")
            and policy.scandot_encoder is not None
        )
        if self._has_scan:
            self.scandot_encoder = copy.deepcopy(policy.scandot_encoder)
            # NOTE: scan_obs_normalizer is NOT applied at inference in ActorCriticRMA.act_inference.
            # Dim: scandot_encoder is an MLP; first Linear's in_features
            self._scan_dim = self.scandot_encoder[0].in_features

        # priv_only fallback: policy has priv_encoder but no history_encoder
        self._has_priv_only = (
            (not self.is_recurrent)
            and (not self._has_history)
            and hasattr(policy, "priv_encoder")
            and policy.priv_encoder is not None
        )
        if self._has_priv_only:
            self.priv_encoder = copy.deepcopy(policy.priv_encoder)
            if hasattr(policy, "priv_obs_normalizer"):
                self.priv_obs_normalizer = copy.deepcopy(policy.priv_obs_normalizer)
            else:
                self.priv_obs_normalizer = nn.Identity()
            self._priv_dim = _infer_dim(
                self.priv_obs_normalizer,
                fallback=self.priv_encoder[0].in_features,
            )

        # Infer proprio dim
        self._proprio_dim = _infer_dim(self.actor_obs_normalizer, fallback=None)
        if self._proprio_dim is None:
            actor_in = self.actor[0].in_features
            encoder_out = 0
            if self._has_history:
                encoder_out += self.history_encoder.linear_output[-2].out_features
            if self._has_scan:
                encoder_out += self.scandot_encoder[-1].out_features
            if self._has_priv_only:
                encoder_out += self.priv_encoder[-1].out_features
            self._proprio_dim = actor_in - encoder_out

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.actor_obs_normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.actor_obs_normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, proprio, *extras):
        """Dynamic forward mirroring ActorCriticRMA.act_inference order:
        proprio_normed -> history_latent -> scan_latent.

        For simple ActorCritic (no encoders), extras is empty and only proprio is used.
        """
        actor_input_parts = [self.actor_obs_normalizer(proprio)]
        extras_idx = 0

        if self._has_history:
            history = extras[extras_idx]
            extras_idx += 1
            history_latent = self.history_encoder(self.history_obs_normalizer(history))
            actor_input_parts.append(history_latent)
        elif self._has_priv_only:
            priv = extras[extras_idx]
            extras_idx += 1
            priv_latent = self.priv_encoder(self.priv_obs_normalizer(priv))
            actor_input_parts.append(priv_latent)

        if self._has_scan:
            scan = extras[extras_idx]
            # No scan normalizer applied here — mirrors act_inference behavior
            scan_latent = self.scandot_encoder(scan)
            actor_input_parts.append(scan_latent)

        return self.actor(torch.cat(actor_input_parts, dim=-1))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.

        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            # Build dynamic dummy inputs + input_names
            dummy_inputs = [torch.zeros(1, self._proprio_dim)]
            input_names = ["proprio"]

            if self._has_history:
                dummy_inputs.append(torch.zeros(1, self._history_dim))
                input_names.append("history")
            elif self._has_priv_only:
                dummy_inputs.append(torch.zeros(1, self._priv_dim))
                input_names.append("priv")

            if self._has_scan:
                dummy_inputs.append(torch.zeros(1, self._scan_dim))
                input_names.append("scan")

            torch.onnx.export(
                self,
                tuple(dummy_inputs),
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=input_names,
                output_names=["actions"],
                dynamic_axes={name: {0: "batch"} for name in input_names + ["actions"]},
            )
