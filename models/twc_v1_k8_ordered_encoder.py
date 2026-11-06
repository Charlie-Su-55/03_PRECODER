"""Ordered-window readout for the existing K8 UE, with exact state migration.

Only the two encoder head input widths change.  The original model remains
available and the decoder/feedback implementation is reused without changes.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.twc_v1_k8_neural import (
    K8SwinEncoder, K8SwinCSIModel, K_USERS, N_ANTENNAS, N_SUBCARRIERS,
    SUPPORTED_GROUP_COUNTS, shared_dimension, angular_delay_transform,
)


ARCHITECTURE = 'twc_v1_k8_ordered_window_encoder_v1'
TOKEN_HEIGHT, TOKEN_WIDTH, TOKEN_CHANNELS = 64, 8, 128
WINDOW_SIZE, WINDOW_COUNT = 4, 32
HEAD_WEIGHT_KEYS = ('encoder.key_head.weight', 'encoder.shared_head.weight')
EXPECTED_PARAMETER_DELTA = (4096 - 128) * (8 + 320)
RTOL, ATOL = 1e-5, 1e-6


def ordered_window_pool(tokens):
    """[BK,512,C] -> [BK,32,C], delay-window-major then angle-window-major.

    C is permitted to vary for small algebra tests; production C is fixed128.
    This pools only the sixteen tokens inside each actual4x4 spatial window.
    """
    if (not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[0] < 1
            or tokens.shape[1] != TOKEN_HEIGHT * TOKEN_WIDTH or tokens.shape[2] < 1
            or not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all())):
        raise ValueError('Finite floating tokens[BK,512,C] are required')
    return tokens.reshape(tokens.shape[0], 16, 4, 2, 4, tokens.shape[2]).mean(
        dim=(2, 4)).reshape(tokens.shape[0], WINDOW_COUNT, tokens.shape[2])


def expand_mean_head_weight(weight):
    """[out,C] -> [out,32*C], with all32 blocks equal to W/32.

    For normalized tokens x_i and group condition c, every window has16 tokens:
      sum_j (W/32)(mean_{i in j}(x_i)+c)+b = W(mean_i(x_i)+c)+b.
    Adding c to every window is necessary; adding it only once after flattening
    would change the original G-dependent contribution.
    """
    if (not torch.is_tensor(weight) or weight.ndim != 2 or min(weight.shape) < 1
            or not weight.is_floating_point() or not bool(torch.isfinite(weight).all())):
        raise ValueError('Finite real head weight[out,C] required')
    return (weight / WINDOW_COUNT).repeat(1, WINDOW_COUNT)


def migrate_state_dict(source_state, target_template):
    """Convert only the two allowed head weights; preserve every other value.

    The template comes from the actual target model.  Generic head dimensions
    support tiny tensor tests without constructing a full codec or channel.
    """
    if (not isinstance(source_state, Mapping) or not isinstance(target_template, Mapping)
            or not source_state or set(source_state) != set(target_template)):
        raise ValueError('Source and target must have the exact same complete state keys')
    if not set(HEAD_WEIGHT_KEYS) <= set(source_state):
        raise ValueError('Both explicitly permitted encoder head keys are required')
    converted = {}
    changed = []
    for name, target in target_template.items():
        value = source_state[name]
        if (not torch.is_tensor(value) or not torch.is_tensor(target)
                or value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Source/target tensor dtype or finite check failed: '+name)
        if name in HEAD_WEIGHT_KEYS:
            if (value.ndim != 2 or target.ndim != 2 or value.shape[0] != target.shape[0]
                    or target.shape[1] != WINDOW_COUNT * value.shape[1]):
                raise ValueError('Only the32-fold head input expansion is allowed: '+name)
            converted[name] = expand_mean_head_weight(value)
            changed.append(name)
        else:
            if value.shape != target.shape:
                raise ValueError('Non-head state shape changed: '+name)
            converted[name] = value
    if set(changed) != set(HEAD_WEIGHT_KEYS):
        raise ValueError('Exactly two encoder head weights must be expanded')
    return converted, dict(
        architecture=ARCHITECTURE, changed_keys=sorted(changed),
        copied_keys=sorted(set(source_state)-set(changed)),
        window_count=WINDOW_COUNT, head_initialization='32 ordered copies of W_old/32; b unchanged',
        group_condition='same original G embedding added to every pooled window',
        source_values_modified=False, strict_loading=True,
        expected_production_parameter_delta=EXPECTED_PARAMETER_DELTA,
        numerical_equivalence='mathematical identity; floating errors must be measured')


def migrate_mean_state(model, source_state):
    """Explicit initialization only; ordinary ordered resume must load strictly."""
    if not isinstance(model, nn.Module):
        raise TypeError('A target torch module is required')
    converted, evidence = migrate_state_dict(source_state, model.state_dict())
    model.load_state_dict(converted, strict=True)
    loaded = model.state_dict()
    for name, expected in converted.items():
        if not torch.equal(loaded[name].detach().cpu(), expected.detach().cpu()):
            raise RuntimeError('Strict migration did not preserve expected value: '+name)
    return evidence


def comparison_error(actual, expected, *, rtol=RTOL, atol=ATOL):
    """Report actual errors; allclose uses atol+rtol*abs(expected), not max_rel."""
    if (not torch.is_tensor(actual) or not torch.is_tensor(expected)
            or actual.shape != expected.shape or actual.dtype != expected.dtype):
        raise ValueError('Matching shape/dtype tensor comparisons required')
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        return dict(passed=False, max_abs=None, max_rel=None, rtol=rtol, atol=atol,
                    error='nonfinite values')
    error = (actual - expected).abs()
    return dict(passed=bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
                max_abs=float(error.max().item()) if error.numel() else 0.,
                max_rel=float((error / expected.abs().clamp_min(1e-12)).max().item()) if error.numel() else 0.,
                rtol=rtol, atol=atol, relative_denominator_floor=1e-12)


class K8OrderedWindowSwinEncoder(K8SwinEncoder):
    """Existing K8 Swin plus ordered means of32 windows and4096-wide heads."""

    def __init__(self, **swin_kwargs):
        frozen=dict(embed_dim=128,blocks=4,patch_size=4,window_size=4,heads=4,mlp_ratio=4.0)
        if set(swin_kwargs)-set(frozen) or any(swin_kwargs.get(k,v)!=v for k,v in frozen.items()):
            raise ValueError('Ordered-window v1 retains the original width/blocks/patch/window/heads/MLP')
        super().__init__(**swin_kwargs)
        width = self.norm.normalized_shape[0]
        if width != TOKEN_CHANNELS or self.patch_size != 4 or self.window_size != 4:
            raise ValueError('Ordered-window v1 fixes width128, patch4 and window4')
        # The replacement is outside forward and preserves original state keys.
        self.key_head = nn.Linear(WINDOW_COUNT * width, 8)
        self.shared_head = nn.Linear(WINDOW_COUNT * width, 320)

    def forward(self, H_dl, *, group_count):
        if type(group_count) is not int or group_count not in SUPPORTED_GROUP_COUNTS:
            raise ValueError('group_count must be1,2,4')
        if (not torch.is_tensor(H_dl) or H_dl.dtype != torch.complex64 or H_dl.ndim != 4
                or tuple(H_dl.shape[1:]) != (K_USERS, N_ANTENNAS, N_SUBCARRIERS)
                or not bool(torch.isfinite(H_dl).all())):
            raise ValueError('H_dl must be finite complex64[B,8,32,256]')
        batch = H_dl.shape[0]
        features = angular_delay_transform(H_dl, N_SUBCARRIERS).reshape(
            batch*K_USERS, 2, N_SUBCARRIERS, N_ANTENNAS)
        tokens, height, width = self.patch_embed(features)
        if (height, width) != (TOKEN_HEIGHT, TOKEN_WIDTH):
            raise ValueError('The actual ordered token grid must be64x8')
        for block in self.blocks:
            tokens = block(tokens, height, width)
        pooled = ordered_window_pool(self.norm(tokens)).reshape(batch, K_USERS, WINDOW_COUNT, TOKEN_CHANNELS)
        index = SUPPORTED_GROUP_COUNTS.index(group_count)
        condition = self.group_embedding(torch.full((batch,), index, device=H_dl.device, dtype=torch.long))
        conditioned = (pooled + condition[:, None, None, :]).flatten(-2)
        key_real = self.key_head(conditioned)
        shared_real = self.shared_head(conditioned)
        key = torch.complex(key_real[..., :4], key_real[..., 4:])
        shared = torch.complex(shared_real[..., :160], shared_real[..., 160:])[..., :shared_dimension(group_count)]
        return key, shared


def build_codec(ordered=False, **swin_kwargs):
    """Build the old codec and replace only encoder; preserve caller RNG state.

    Full construction is reserved for server runtime.  Local tests call only
    the pooling and migration algebra helpers above, never this factory.
    """
    if type(ordered) is not bool:
        raise TypeError('ordered must be an explicit bool')
    model = K8SwinCSIModel(**swin_kwargs)
    if ordered:
        # Existing decoder initialization is identical to mean-codec creation.
        # Extra encoder construction consumes no external CPU/CUDA RNG stream.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            model.encoder = K8OrderedWindowSwinEncoder(**swin_kwargs)
    return model
