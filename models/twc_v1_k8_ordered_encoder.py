"""Numerically stable ordered-window contrasts beside the unchanged K8 mean path.

The old128-wide key/shared heads remain exactly intact.  Two zero-initialized,
bias-free window-difference heads add the remaining ordered spatial degrees of
freedom without replacing the source model's mean reduction or GEMM shapes.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.twc_v1_k8_neural import (
    K8SwinEncoder, K8SwinCSIModel, K_USERS, N_ANTENNAS, N_SUBCARRIERS,
    SUPPORTED_GROUP_COUNTS, shared_dimension, angular_delay_transform,
)


ARCHITECTURE = 'twc_v1_k8_ordered_window_encoder_v2'
TOKEN_HEIGHT, TOKEN_WIDTH, TOKEN_CHANNELS = 64, 8, 128
WINDOW_SIZE, WINDOW_COUNT = 4, 32
HEAD_WEIGHT_KEYS = ('encoder.key_head.weight', 'encoder.shared_head.weight')
RESIDUAL_WEIGHT_KEYS = ('encoder.key_window_residual.weight', 'encoder.shared_window_residual.weight')
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


def window_difference_features(tokens):
    """Ordered concat(p1-p32,...,p31-p32), with no G embedding mixed in."""
    windows = ordered_window_pool(tokens)
    differences = (windows[:, :-1] - windows[:, -1:]).flatten(1)
    if not bool(torch.isfinite(differences).all()):
        raise FloatingPointError('Nonfinite ordered window differences')
    return differences


def readout_from_normalized_tokens(tokens, group_condition, key_head, shared_head,
                                   key_window_residual, shared_window_residual, *, users=K_USERS):
    """The production readout, also usable with tiny real tensor/module tests.

    tokens is the single original LayerNorm result[B*K,512,C].  The old
    mean/reshape/G-add/Linear sequence is evaluated with its original shapes;
    differences are computed from the same tokens and are always in the graph.
    """
    if (not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[1] != 512
            or type(users) is not int or users < 1 or tokens.shape[0] % users
            or not torch.is_tensor(group_condition) or group_condition.ndim != 2
            or tuple(group_condition.shape) != (tokens.shape[0]//users,tokens.shape[2])
            or tokens.dtype != group_condition.dtype or tokens.device != group_condition.device):
        raise ValueError('Matched normalized tokens and group condition shapes/dtypes/devices required')
    batch, channels = group_condition.shape
    pooled = tokens.mean(dim=1).reshape(batch, users, channels)
    conditioned = pooled + group_condition[:, None]
    key = key_head(conditioned)
    shared = shared_head(conditioned)
    differences = window_difference_features(tokens).reshape(batch, users, (WINDOW_COUNT-1)*channels)
    return key + key_window_residual(differences), shared + shared_window_residual(differences)


def ordered_head_coordinates(weight):
    """Algebraic coordinates of an arbitrary32-window linear head.

    This is a capacity-identity helper, never the source initialization path.
    For window weights A_j, W=sum_j(A_j), R_j=A_j-W/32 (j<32).  Then
    W(mean(p)+c)+sum_j R_j(p_j-p32) = sum_j A_j(p_j+c).
    """
    if (not torch.is_tensor(weight) or weight.ndim != 2 or min(weight.shape) < 1
            or weight.shape[1] % WINDOW_COUNT or not weight.is_floating_point()
            or not bool(torch.isfinite(weight).all())):
        raise ValueError('Finite real ordered head weight[out,32*C] required')
    blocks = weight.reshape(weight.shape[0], WINDOW_COUNT, weight.shape[1]//WINDOW_COUNT)
    mean_weight = blocks.sum(dim=1)
    residual_weight = (blocks[:, :-1] - mean_weight[:, None]/WINDOW_COUNT).flatten(1)
    return mean_weight, residual_weight


def migrate_state_dict(source_state, target_template):
    """Copy the entire old state unchanged; add only two zero residual weights.

    Generic channel/head dimensions permit tiny tests.  Actual target shapes
    come from the real v2 codec; no keys are dropped, filtered or expanded.
    """
    if (not isinstance(source_state, Mapping) or not isinstance(target_template, Mapping)
            or not source_state
            or set(target_template)-set(source_state) != set(RESIDUAL_WEIGHT_KEYS)
            or set(source_state)-set(target_template)):
        raise ValueError('Exact complete old keys plus only two new residual weights required')
    if not set(HEAD_WEIGHT_KEYS) <= set(source_state):
        raise ValueError('Both explicitly permitted encoder head keys are required')
    converted = {}
    for name, target in target_template.items():
        if not torch.is_tensor(target):raise ValueError('Target state must contain tensors: '+name)
        if name in RESIDUAL_WEIGHT_KEYS:
            old_name = HEAD_WEIGHT_KEYS[RESIDUAL_WEIGHT_KEYS.index(name)]
            head = source_state[old_name]
            if (not torch.is_tensor(head) or head.ndim != 2 or target.ndim != 2
                    or target.shape != (head.shape[0],(WINDOW_COUNT-1)*head.shape[1])
                    or target.dtype != head.dtype or not target.is_floating_point()):
                raise ValueError('Residual shape/dtype differs from the old head: '+name)
            converted[name] = torch.zeros_like(target)
            continue
        value = source_state[name]
        if (not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Old state shape/dtype/finite check failed: '+name)
        converted[name] = value
    return converted, dict(
        architecture=ARCHITECTURE, changed_keys=[],new_keys=sorted(RESIDUAL_WEIGHT_KEYS),
        new_parameter_keys=sorted(RESIDUAL_WEIGHT_KEYS),copied_keys=sorted(source_state),
        window_count=WINDOW_COUNT, head_initialization='old heads unchanged; bias-free residual weights zero',
        group_condition='original mean path only; absent from window differences',
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
    """Existing K8 mean readout plus zero-initialized31-window contrasts."""

    def __init__(self, **swin_kwargs):
        frozen=dict(embed_dim=128,blocks=4,patch_size=4,window_size=4,heads=4,mlp_ratio=4.0)
        if set(swin_kwargs)-set(frozen) or any(swin_kwargs.get(k,v)!=v for k,v in frozen.items()):
            raise ValueError('Ordered-window v2 retains the original width/blocks/patch/window/heads/MLP')
        super().__init__(**swin_kwargs)
        width = self.norm.normalized_shape[0]
        if width != TOKEN_CHANNELS or self.patch_size != 4 or self.window_size != 4:
            raise ValueError('Ordered-window v2 fixes width128, patch4 and window4')
        self.key_window_residual = nn.Linear((WINDOW_COUNT-1)*width,8,bias=False)
        self.shared_window_residual = nn.Linear((WINDOW_COUNT-1)*width,320,bias=False)
        nn.init.zeros_(self.key_window_residual.weight)
        nn.init.zeros_(self.shared_window_residual.weight)

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
        normalized_tokens = self.norm(tokens)
        index = SUPPORTED_GROUP_COUNTS.index(group_count)
        condition = self.group_embedding(torch.full((batch,), index, device=H_dl.device, dtype=torch.long))
        key_real,shared_real = readout_from_normalized_tokens(normalized_tokens,condition,
            self.key_head,self.shared_head,self.key_window_residual,self.shared_window_residual)
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
