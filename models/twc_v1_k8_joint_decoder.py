"""K8 Joint Transformer and spatial-Swin BS feedback receiver."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.baseline_common import inverse_angular_delay_transform


K_USERS, N_ANTENNAS, N_SUBCARRIERS = 8, 32, 256
MAX_SYMBOLS_PER_USER = 164
WIDTH, HEADS, QUERY_CHUNK = 128, 4, 64


def _coordinates(position, extent):
    scaled = position.float() / max(extent - 1, 1)
    return torch.stack((scaled, torch.sin(2 * math.pi * scaled),
                        torch.cos(2 * math.pi * scaled)), dim=-1)


def _run(function, *args, enabled):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def _channel_coupling(H_ul_est):
    h = H_ul_est.permute(0, 3, 1, 2)  # [B,256,8,32]
    gram = h @ h.conj().transpose(-2, -1)
    norm2 = h.abs().square().sum(dim=-1).clamp_min(1e-12)
    return (gram.abs().square() /
            (norm2.unsqueeze(-1) * norm2.unsqueeze(-2))).mean(dim=1)


class _ChunkedAttention(nn.Module):
    """Global SDPA with all keys and bounded query chunks."""
    def __init__(self, *, width=WIDTH, heads=HEADS, query_chunk=QUERY_CHUNK,
                 activation_checkpoint=True):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by its head count")
        self.width, self.heads = width, heads
        self.query_chunk = query_chunk
        self.activation_checkpoint = activation_checkpoint
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def _split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads,
                         self.width // self.heads).transpose(1, 2)

    def _attend(self, query, key, value, bias):
        q = self._split(self.q(query))
        attended = F.scaled_dot_product_attention(q, key, value, attn_mask=bias,
            dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.width)
        return self.out(attended)

    def forward(self, query, memory, *, bias_builder=None, key_mask=None):
        key, value = self._split(self.k(memory)), self._split(self.v(memory))
        if key_mask is not None:
            key_mask = key_mask.to(device=query.device, dtype=torch.bool)
        chunks = []
        for start in range(0, query.shape[1], self.query_chunk):
            end = min(start + self.query_chunk, query.shape[1])
            bias = None if bias_builder is None else bias_builder(start, end)
            if key_mask is not None:
                mask_bias = torch.zeros((query.shape[0], 1, 1, memory.shape[1]),
                    dtype=query.dtype, device=query.device)
                mask_bias = mask_bias.masked_fill(~key_mask[:, None, None, :], -torch.inf)
                bias = mask_bias if bias is None else bias + mask_bias
            chunks.append(_run(self._attend, query[:, start:end], key, value, bias,
                enabled=self.activation_checkpoint and self.training))
        return torch.cat(chunks, dim=1)


class _JointLayer(nn.Module):
    def __init__(self, *, width, query_chunk, activation_checkpoint):
        super().__init__()
        self.width = width
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = _ChunkedAttention(width=width, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpoint)
        self.cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = _ChunkedAttention(width=width, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpoint)
        hidden = width * 2
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x, memory, active, self_bias, observation_bias):
        x = x + self.self_attention(self.self_norm(x), self.self_norm(x),
            bias_builder=self_bias, key_mask=active)
        x = x * active[..., None]
        x = x + self.cross_attention(self.cross_norm(x), self.memory_norm(memory),
            bias_builder=observation_bias)
        x = x * active[..., None]
        return (x + self.mlp(self.mlp_norm(x))) * active[..., None]


class K8JointTransformerSwinReceiver(nn.Module):
    """All-user Joint Transformer receiver followed by spatial Swin CSI decoding.

    The public interface deliberately accepts only soft recovered data,
    reliability, noisy received feedback, estimated UL CSI, layout and public
    operating conditions. Simulator truth is not accepted here.
    """
    def __init__(self, base_decoder, *, joint_layers=2, query_chunk=QUERY_CHUNK,
                 activation_checkpoint=True):
        super().__init__()
        self.base_decoder = base_decoder
        self.width = int(base_decoder.embed_dim)
        if self.width % HEADS:
            raise ValueError("Base Swin width must be divisible by four attention heads")
        self.query_chunk = query_chunk
        self.activation_checkpoint = activation_checkpoint
        self.observation_y = nn.Linear(2 * N_ANTENNAS, self.width)
        self.observation_channel = nn.Linear(2 * K_USERS * N_ANTENNAS, self.width)
        self.observation_position = nn.Linear(3, self.width)
        self.observation_layout = nn.Linear(K_USERS * 4, self.width)
        self.observation_reliability = nn.Linear(K_USERS, self.width)
        self.observation_condition = nn.Linear(8, self.width)
        self.observation_norm = nn.LayerNorm(self.width)
        self.query_projection = nn.Linear(23, self.width)
        self.joint_layers = nn.ModuleList([_JointLayer(width=self.width, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpoint) for _ in range(joint_layers)])
        # Per-head biases: same user, same sharing group, same physical RE,
        # estimated UL channel coupling; observation attention uses same-RE bias.
        self.self_bias_weights = nn.Parameter(torch.full((HEADS, 4), 0.02))
        self.observation_bias_weights = nn.Parameter(torch.full((HEADS,), 0.02))
        self.residual_norm = nn.LayerNorm(self.width)
        self.residual_head = nn.Linear(self.width, 2)
        nn.init.normal_(self.residual_head.weight, std=0.001)
        nn.init.zeros_(self.residual_head.bias)
        self.spatial_position = nn.Linear(14, self.width)
        self.spatial_query_norm = nn.LayerNorm(self.width)
        self.spatial_context_norm = nn.LayerNorm(self.width)
        self.spatial_attention = _ChunkedAttention(width=self.width, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpoint)
        nn.init.normal_(self.spatial_attention.out.weight, std=0.001)
        nn.init.zeros_(self.spatial_attention.out.bias)
        self.register_buffer("resource_coordinates", _coordinates(torch.arange(N_SUBCARRIERS),
            N_SUBCARRIERS), persistent=False)
        rows = _coordinates(torch.arange(64), 64)
        cols = _coordinates(torch.arange(8), 8)
        users = torch.eye(K_USERS)
        spatial = torch.cat((rows[None, :, None].expand(K_USERS, 64, 8, 3),
            cols[None, None, :].expand(K_USERS, 64, 8, 3),
            users[:, None, None, :].expand(K_USERS, 64, 8, K_USERS)), dim=-1)
        self.register_buffer("spatial_features", spatial.reshape(K_USERS, 512, 14),
            persistent=False)
        self.register_buffer("spatial_user_ids", torch.arange(K_USERS).repeat_interleave(512),
            persistent=False)

    @staticmethod
    def _scalar(value, batch, device, name, *, allow_zero=True):
        result = torch.as_tensor(value, dtype=torch.float32, device=device)
        if result.ndim == 0:
            result = result.expand(batch)
        valid_sign = result >= 0 if allow_zero else result > 0
        if result.shape != (batch,) or not bool(torch.isfinite(result).all() & valid_sign.all()):
            raise ValueError(f"{name} must be a finite valid scalar or [B]")
        return result

    @staticmethod
    def _slot_map(layout):
        resource = torch.zeros((K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.long)
        kind = torch.zeros_like(resource)  # 0 key, 1 shared, 2 FDMA data
        group_id = torch.full_like(resource, -1)
        active = torch.zeros((K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.bool)
        if hasattr(layout, "shared_positions_by_group"):
            groups = layout.groups
            user_group = {user: gi for gi, members in enumerate(groups) for user in members}
            for user, positions in enumerate(layout.key_positions):
                resource[user, :4] = torch.as_tensor(positions)
                group_id[user, :4] = user_group[user]
                active[user, :4] = True
            width = layout.shared_dimension
            for gi, (members, positions) in enumerate(zip(groups, layout.shared_positions_by_group)):
                for user in members:
                    resource[user, 4:4 + width] = torch.as_tensor(positions)
                    kind[user, 4:4 + width] = 1
                    group_id[user, 4:4 + width] = gi
                    active[user, 4:4 + width] = True
            return resource, kind, group_id, active, "hybrid", layout.group_count
        if hasattr(layout, "data_positions"):
            for user, positions in enumerate(layout.data_positions):
                resource[user, :24] = torch.as_tensor(positions)
                kind[user, :24] = 2
                group_id[user, :24] = -1
                active[user, :24] = True
            return resource, kind, group_id, active, "fdma", 1
        raise TypeError("layout must be a K8 HybridLayout or FDMALayout")

    def _observation_memory(self, Y_rx, H_ul_est, reliability_grid, role_grid,
                            gamma, n0, ep, ek, es, group_count, layout_kind, ce_var):
        batch = Y_rx.shape[0]
        y = torch.view_as_real(Y_rx.transpose(1, 2).contiguous()).flatten(-2)
        h = torch.view_as_real(H_ul_est.permute(0, 3, 1, 2).contiguous()).flatten(-3)
        conditions = torch.stack((gamma / 25.0, torch.log10(n0), ep / 64.0,
            ek / 64.0, es / 64.0, torch.full_like(gamma, group_count / 4.0),
            torch.full_like(gamma, float(layout_kind == "fdma")),
            torch.log10(ce_var.mean(dim=1).clamp_min(1e-12))), dim=-1)
        memory = (self.observation_y(y) + self.observation_channel(h)
            + self.observation_position(self.resource_coordinates).unsqueeze(0)
            + self.observation_layout(role_grid)
            + self.observation_reliability(reliability_grid)
            + self.observation_condition(conditions)[:, None])
        return self.observation_norm(memory)

    def _query_metadata(self, resource, kind, group, active, reliability, ce_var,
                        batch, device):
        resource = resource.to(device)
        kind = kind.to(device)
        group = group.to(device)
        active = active.to(device)
        users = torch.arange(K_USERS, device=device)[:, None].expand(-1, MAX_SYMBOLS_PER_USER)
        user_onehot = F.one_hot(users, K_USERS).float()
        kind_onehot = F.one_hot(kind, 3).float()
        group_safe = group.clamp_min(0)
        group_onehot = F.one_hot(group_safe, 4).float() * active[..., None]
        local = torch.arange(MAX_SYMBOLS_PER_USER, device=device).float()[None].expand(K_USERS, -1)
        local = local / (MAX_SYMBOLS_PER_USER - 1)
        coordinates = self.resource_coordinates[resource]
        reliability_feature = torch.log1p(reliability.clamp_min(0))[..., None]
        ce_feature = torch.log10(ce_var.clamp_min(1e-12))[:, :, None, None].expand(
            -1, -1, MAX_SYMBOLS_PER_USER, -1)
        static = torch.cat((coordinates[None].expand(batch, -1, -1, -1),
            local[None, :, :, None].expand(batch, -1, -1, -1),
            user_onehot[None].expand(batch, -1, -1, -1),
            kind_onehot[None].expand(batch, -1, -1, -1),
            group_onehot[None].expand(batch, -1, -1, -1), ce_feature), dim=-1)
        # Static feature width is 3+1+8+3+4+1 = 20; complex symbol and
        # reliability make the complete projection width 23.
        return static, users.flatten(), group.flatten(), resource.flatten(), active.flatten()

    def _self_bias(self, user_ids, group_ids, resource_ids, active, coupling):
        def build(start, end):
            q_user = user_ids[start:end]
            k_user = user_ids
            q_group, k_group = group_ids[start:end], group_ids
            same_user = (q_user[:, None] == k_user[None]).float()
            same_group = ((q_group[:, None] == k_group[None]) & (q_group[:, None] >= 0)).float()
            same_resource = (resource_ids[start:end, None] == resource_ids[None]).float()
            channel = coupling[:, q_user[:, None], k_user[None, :]]
            terms = (same_user[None, None] * self.self_bias_weights[None, :, 0, None, None]
                + same_group[None, None] * self.self_bias_weights[None, :, 1, None, None]
                + same_resource[None, None] * self.self_bias_weights[None, :, 2, None, None]
                + channel[:, None] * self.self_bias_weights[None, :, 3, None, None])
            return terms
        # Query sequence is user-major and metadata flattened in that same order.
        return build

    def _observation_bias(self, resource_ids):
        def build(start, end):
            same = (resource_ids[start:end, None] ==
                    torch.arange(N_SUBCARRIERS, device=resource_ids.device)[None]).float()
            return same[None, None] * self.observation_bias_weights[None, :, None, None]
        return build

    def forward(self, key, shared, *, group_count, Y_rx, H_ul_est, layout,
                feedback_gamma_db, noise_variance, pilot_energy_per_ue,
                key_energy_per_ue, shared_energy_per_ue, channel_error_variance,
                key_reliability, shared_reliability):
        resource, kind, group, active, layout_kind, mapped_group_count = self._slot_map(layout)
        if group_count != mapped_group_count:
            raise ValueError("group_count and layout disagree")
        batch = key.shape[0] if torch.is_tensor(key) and key.ndim else 0
        if key.dtype != torch.complex64 or tuple(key.shape) != (batch, K_USERS, 4):
            raise ValueError("key must be complex64 [B,8,4]")
        expected_shared = 160 if layout_kind == "fdma" else int(active[0, 4:].sum())
        if shared.dtype != torch.complex64 or tuple(shared.shape) != (batch, K_USERS, expected_shared):
            raise ValueError(f"shared must be complex64 [B,8,{expected_shared}]")
        if Y_rx.dtype != torch.complex64 or tuple(Y_rx.shape) != (batch, N_ANTENNAS, N_SUBCARRIERS):
            raise ValueError("Y_rx must be complex64 [B,32,256]")
        if H_ul_est.dtype != torch.complex64 or tuple(H_ul_est.shape) != (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS):
            raise ValueError("H_ul_est must be complex64 [B,8,32,256]")
        device = self.base_decoder.fc.weight.device
        if any(t.device != device for t in (key, shared, Y_rx, H_ul_est)):
            raise ValueError("Receiver inputs and parameters must share a device")
        if not all(bool(torch.isfinite(t).all()) for t in (key, shared, Y_rx, H_ul_est)):
            raise ValueError("Receiver complex inputs must be finite")
        gamma = self._scalar(feedback_gamma_db, batch, device, "feedback_gamma_db")
        n0 = self._scalar(noise_variance, batch, device, "noise_variance", allow_zero=False)
        ep = self._scalar(pilot_energy_per_ue, batch, device, "pilot_energy_per_ue", allow_zero=False)
        ek = self._scalar(key_energy_per_ue, batch, device, "key_energy_per_ue")
        es = self._scalar(shared_energy_per_ue, batch, device, "shared_energy_per_ue", allow_zero=False)
        ce_var = torch.as_tensor(channel_error_variance, dtype=torch.float32, device=device)
        if ce_var.ndim == 0:
            ce_var = ce_var.expand(batch, K_USERS)
        if ce_var.shape != (batch, K_USERS) or not bool(torch.isfinite(ce_var).all() & (ce_var >= 0).all()):
            raise ValueError("channel_error_variance must be finite nonnegative [B,8]")
        if not bool(((gamma >= 0) & (gamma <= 25)).all()):
            raise ValueError("feedback_gamma_db must lie in [0,25]")

        key_rel = torch.as_tensor(key_reliability, dtype=torch.float32, device=device)
        shared_rel = torch.as_tensor(shared_reliability, dtype=torch.float32, device=device)
        active_shared = 20 if layout_kind == "fdma" else expected_shared
        if key_rel.shape != (batch, K_USERS, 4) or shared_rel.shape != (batch, K_USERS, active_shared):
            raise ValueError("Reliability shape does not match active received symbols")
        if not bool(torch.isfinite(key_rel).all() & torch.isfinite(shared_rel).all()
                    & (key_rel >= 0).all() & (key_rel <= 1).all()
                    & (shared_rel >= 0).all() & (shared_rel <= 1).all()):
            raise ValueError("Reliabilities must be finite and lie in [0,1]")

        symbols = torch.zeros((batch, K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.complex64, device=device)
        reliability = torch.zeros((batch, K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.float32, device=device)
        mask = active.to(device)[None].expand(batch, -1, -1)
        symbols[..., :4] = key
        reliability[..., :4] = key_rel
        if layout_kind == "fdma":
            symbols[..., 4:24] = shared[..., :20]
            reliability[..., 4:24] = shared_rel
            # The first four packed FDMA coordinates are passed through key;
            # relabel every active coordinate as FDMA data in query metadata.
            kind = torch.zeros_like(kind)
            kind[:, :24] = 2
        else:
            symbols[..., 4:4 + expected_shared] = shared
            reliability[..., 4:4 + expected_shared] = shared_rel

        resource_d = resource.to(device)
        role_grid_cpu = torch.zeros((N_SUBCARRIERS, K_USERS * 4), dtype=torch.float32)
        for user, positions in enumerate(layout.pilot_positions):
            role_grid_cpu[torch.as_tensor(positions), user * 4] = 1.0
        rel_grid = torch.zeros((batch, N_SUBCARRIERS, K_USERS), dtype=torch.float32, device=device)
        for user in range(K_USERS):
            slots = torch.nonzero(active[user], as_tuple=False).flatten()
            q_cpu = resource[user, slots]
            role_grid_cpu[q_cpu, user * 4 + kind[user, slots] + 1] = 1.0
            q = resource_d[user, slots.to(device)]
            rel_grid[:, q, user] = reliability[:, user, slots.to(device)]
        role_grid = role_grid_cpu.to(device)
        memory = self._observation_memory(Y_rx, H_ul_est, rel_grid, role_grid,
            gamma, n0, ep, ek, es, group_count, layout_kind, ce_var)
        static, user_ids, group_ids, resource_ids, flat_active = self._query_metadata(
            resource, kind, group, active, reliability, ce_var, batch, device)
        features = torch.cat((torch.view_as_real(symbols.contiguous()),
            torch.log1p(reliability)[..., None], static), dim=-1).reshape(
                batch, K_USERS * MAX_SYMBOLS_PER_USER, 23)
        context = self.query_projection(features) * flat_active[None, :, None]
        coupling = _channel_coupling(H_ul_est)
        self_bias = self._self_bias(user_ids, group_ids, resource_ids, flat_active, coupling)
        obs_bias = self._observation_bias(resource_ids)
        for layer in self.joint_layers:
            context = _run(layer, context, memory, flat_active[None].expand(batch, -1),
                self_bias, obs_bias,
                enabled=self.activation_checkpoint and self.training)
        context_by_user = context.reshape(batch, K_USERS, MAX_SYMBOLS_PER_USER, self.width)
        residual = self.residual_head(self.residual_norm(context_by_user))
        correction = torch.complex(residual[..., 0], residual[..., 1]).masked_fill(~mask, 0)
        corrected = (symbols + correction).masked_fill(~mask, 0)

        spatial = self.base_decoder.expand_packed_symbols(corrected, mask,
            group_count=group_count, layout_kind=layout_kind)
        spatial = spatial.reshape(batch, K_USERS, 512, self.width)
        position = self.spatial_position(self.spatial_features).unsqueeze(0)
        spatial = spatial + position
        spatial_flat = spatial.reshape(batch, K_USERS * 512, self.width)
        context_keys = self.spatial_context_norm(context)
        spatial_flat = spatial_flat + self.spatial_attention(
            self.spatial_query_norm(spatial_flat), context_keys,
            key_mask=flat_active[None].expand(batch, -1))
        spatial = spatial_flat.reshape(batch * K_USERS, 512, self.width)
        for block in self.base_decoder.blocks:
            spatial = _run(block, spatial, 64, 8,
                enabled=self.activation_checkpoint and self.training)
        H_hat = self.base_decoder.tokens_to_channel(spatial, batch, normalize=True)
        if not bool(torch.isfinite(H_hat).all() & torch.isfinite(corrected).all()):
            raise FloatingPointError("K8 Joint Transformer receiver produced nonfinite output")
        execution = {
            "layout": layout_kind, "group_count": group_count,
            "active_query_count": int(active.sum()), "padded_query_count": int(active.numel() - active.sum()),
            "observation_memory_shape": list(memory.shape),
            "joint_context_shape": list(context.shape),
            "spatial_cross_attention_shape": [batch, K_USERS * 512, self.width],
            "swin_block_count": len(self.base_decoder.blocks),
            "estimated_channel_coupling_shape": list(coupling.shape),
        }
        return H_hat, corrected, execution
