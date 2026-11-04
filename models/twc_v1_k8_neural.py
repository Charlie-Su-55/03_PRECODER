"""K8 models adapted from the read-only 08 pilot-U-Net/Swin designs.

The module includes a pilot-only CE model, UE Swin encoder/CSI decoder, and a
multi-source conditioned Swin receiver prototype. The latter is not yet the
planned full Joint Transformer plus spatial-Swin architecture. Training,
channel simulation, and precoder design live in separate modules.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from models.baseline_common import angular_delay_transform, inverse_angular_delay_transform
from models.swin_precoder import PatchEmbedding, SwinTransformerBlock


_PROTOCOL_NAME = "_twc_v1_k8_protocol"
_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "utils" / "twc_v1_k8_protocol.py"
if _PROTOCOL_NAME in sys.modules:
    _protocol = sys.modules[_PROTOCOL_NAME]
    if Path(_protocol.__file__).resolve() != _PROTOCOL_PATH:
        raise RuntimeError("TWC K8 protocol module identity collision")
else:
    _spec = importlib.util.spec_from_file_location(_PROTOCOL_NAME, _PROTOCOL_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Cannot load TWC K8 protocol module")
    _protocol = importlib.util.module_from_spec(_spec)
    sys.modules[_PROTOCOL_NAME] = _protocol
    try:
        _spec.loader.exec_module(_protocol)
    except BaseException:
        del sys.modules[_PROTOCOL_NAME]
        raise

K_USERS = _protocol.K_USERS
N_ANTENNAS = _protocol.N_ANTENNAS
N_SUBCARRIERS = _protocol.N_SUBCARRIERS
PILOT_POSITIONS_BY_USER = _protocol.PILOT_POSITIONS_BY_USER
SUPPORTED_GROUP_COUNTS = _protocol.SUPPORTED_GROUP_COUNTS
shared_dimension = _protocol.shared_dimension


def _complex(value, name, shape):
    if (not torch.is_tensor(value) or value.dtype != torch.complex64
            or tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite complex64 with shape {shape}")


def _groups(group_count):
    if type(group_count) is not int or group_count not in SUPPORTED_GROUP_COUNTS:
        raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")


class _UNetBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(inputs, outputs, 5, padding=2), nn.GroupNorm(8, outputs), nn.GELU(),
            nn.Conv1d(outputs, outputs, 5, padding=2), nn.GroupNorm(8, outputs), nn.GELU(),
        )

    def forward(self, value):
        return self.layers(value)


class K8PilotResidualUNet(nn.Module):
    """One UE-shared BS frequency U-Net; only the 64 pilot RE of Y are read."""

    feature_channels = 4 * N_ANTENNAS + 6

    def __init__(self):
        super().__init__()
        mask = torch.zeros(K_USERS, N_SUBCARRIERS, dtype=torch.bool)
        for user, positions in enumerate(PILOT_POSITIONS_BY_USER):
            mask[user, list(positions)] = True
        self.register_buffer("pilot_mask", mask, persistent=True)
        self.register_buffer("pilot_union", mask.any(0), persistent=True)
        self.encoder0 = _UNetBlock(self.feature_channels, 32)
        self.down1 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.encoder1 = _UNetBlock(64, 64)
        self.down2 = nn.Conv1d(64, 96, 5, stride=2, padding=2)
        self.bottleneck = _UNetBlock(96, 96)
        self.decoder1 = _UNetBlock(96 + 64, 64)
        self.decoder0 = _UNetBlock(64 + 32, 32)
        self.head = nn.Conv1d(32, 2 * N_ANTENNAS, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _noise(noise_variance, batch, device):
        value = torch.as_tensor(noise_variance, dtype=torch.float32, device=device)
        if value.ndim == 0:
            value = value.expand(batch)
        if (value.shape != (batch,) or not bool(torch.isfinite(value).all())
                or not bool((value > 0).all())):
            raise ValueError("noise_variance must be positive finite scalar or [B]")
        # TWC K8 definition: reference energy 64 divided by 256 RE.
        gamma = -10.0 * torch.log10(value / (64.0 / 256.0))
        if not bool(((gamma >= 0.0) & (gamma <= 25.0)).all()):
            raise ValueError("K8 U-Net noise variance must map to feedback gamma in [0,25] dB")
        return value, ((gamma - 12.5) / 12.5).clamp(-1.0, 1.0)

    def forward(self, Y, pilots, H_initial, *, pilot_energy_per_ue, noise_variance):
        batch = Y.shape[0] if torch.is_tensor(Y) and Y.ndim else 0
        if (not torch.is_tensor(Y) or Y.dtype != torch.complex64
                or tuple(Y.shape) != (batch, N_ANTENNAS, N_SUBCARRIERS)):
            raise ValueError(f"Y must be complex64 with shape {(batch, N_ANTENNAS, N_SUBCARRIERS)}")
        _complex(pilots, "pilots", (batch, K_USERS, N_SUBCARRIERS))
        _complex(H_initial, "H_initial", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
        if len({Y.device, pilots.device, H_initial.device, self.head.weight.device}) != 1:
            raise ValueError("U-Net, pilots and observations must share a device")
        own = self.pilot_mask.to(Y.device)
        union = self.pilot_union.to(Y.device)
        pilot_Y = torch.where(union[None, None], Y, torch.zeros((), dtype=Y.dtype, device=Y.device))
        if (not bool(torch.isfinite(pilot_Y).all())
                or not bool(((pilots[:, ~own] == 0)).all())
                or not bool((torch.isclose(pilots[:, own].abs(), torch.ones_like(pilots[:, own].real),
                                           atol=1e-5, rtol=1e-5)).all())):
            raise ValueError("Known pilots must be unit-modulus on each UE's own pilot REs and zero elsewhere")
        _, snr = self._noise(noise_variance, batch, Y.device)
        pilot_energy = torch.as_tensor(pilot_energy_per_ue, dtype=torch.float32, device=Y.device)
        if pilot_energy.ndim == 0:
            pilot_energy = pilot_energy.expand(batch)
        if (pilot_energy.shape != (batch,) or not bool(torch.isfinite(pilot_energy).all())
                or not bool(((pilot_energy > 0) & (pilot_energy < 64)).all())):
            raise ValueError("pilot_energy_per_ue must be positive finite scalar or [B] and below 64")
        # Ep candidates 8/16/24 map to -1/0/+1; energy is explicit context,
        # not inferred from a noisy pilot observation.
        ep_feature = ((pilot_energy - 16.0) / 8.0).clamp(-1.0, 1.0)
        common = torch.cat((pilot_Y.real, pilot_Y.imag), dim=1)[:, None].expand(-1, K_USERS, -1, -1)
        initial = torch.cat((H_initial.real, H_initial.imag), dim=2)
        features = torch.cat((
            common, initial,
            pilots.real[:, :, None], pilots.imag[:, :, None],
            own[None, :, None].expand(batch, -1, -1, -1),
            union[None, None, None].expand(batch, K_USERS, -1, -1),
            snr[:, None, None, None].expand(-1, K_USERS, 1, N_SUBCARRIERS),
            ep_feature[:, None, None, None].expand(-1, K_USERS, 1, N_SUBCARRIERS),
        ), dim=2).reshape(batch * K_USERS, self.feature_channels, N_SUBCARRIERS)
        x0 = self.encoder0(features)
        x1 = self.encoder1(self.down1(x0))
        bottom = self.bottleneck(self.down2(x1))
        up1 = self.decoder1(torch.cat((F.interpolate(bottom, size=x1.shape[-1], mode="linear",
                                                      align_corners=False), x1), dim=1))
        up0 = self.decoder0(torch.cat((F.interpolate(up1, size=x0.shape[-1], mode="linear",
                                                      align_corners=False), x0), dim=1))
        residual = self.head(up0).reshape(batch, K_USERS, 2 * N_ANTENNAS, N_SUBCARRIERS)
        corrected = H_initial + torch.complex(residual[:, :, :N_ANTENNAS],
                                               residual[:, :, N_ANTENNAS:])
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("K8 U-Net produced a nonfinite channel estimate")
        return corrected


class K8SwinEncoder(nn.Module):
    """UE-shared Swin encoder with fixed maximum key/shared heads for G=1/2/4."""

    def __init__(self, *, embed_dim=128, blocks=4, patch_size=4, window_size=4,
                 heads=4, mlp_ratio=4.0):
        super().__init__()
        if (N_SUBCARRIERS % (patch_size * window_size)
                or N_ANTENNAS % (patch_size * window_size) or embed_dim % heads):
            raise ValueError("Swin patch/window geometry or attention heads are incompatible")
        self.patch_size, self.window_size = patch_size, window_size
        self.patch_embed = PatchEmbedding(2, embed_dim, patch_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, heads, window_size, mlp_ratio) for _ in range(blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.group_embedding = nn.Embedding(3, embed_dim)
        self.key_head = nn.Linear(embed_dim, 2 * 4)
        self.shared_head = nn.Linear(embed_dim, 2 * shared_dimension(1))

    def forward(self, H_dl, *, group_count):
        _groups(group_count)
        if (not torch.is_tensor(H_dl) or H_dl.dtype != torch.complex64 or H_dl.ndim != 4
                or tuple(H_dl.shape[1:]) != (K_USERS, N_ANTENNAS, N_SUBCARRIERS)
                or not bool(torch.isfinite(H_dl).all())):
            raise ValueError("H_dl must be finite complex64 [B,8,32,256]")
        batch = H_dl.shape[0]
        features = angular_delay_transform(H_dl, N_SUBCARRIERS).reshape(
            batch * K_USERS, 2, N_SUBCARRIERS, N_ANTENNAS)
        tokens, height, width = self.patch_embed(features)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        pooled = self.norm(tokens).mean(dim=1).reshape(batch, K_USERS, -1)
        group_index = SUPPORTED_GROUP_COUNTS.index(group_count)
        condition = self.group_embedding(torch.full((batch,), group_index,
                                                    device=H_dl.device, dtype=torch.long))
        conditioned = pooled + condition[:, None]
        key_real = self.key_head(conditioned)
        shared_real = self.shared_head(conditioned)
        key = torch.complex(key_real[..., :4], key_real[..., 4:])
        active = shared_dimension(group_count)
        shared = torch.complex(shared_real[..., :160], shared_real[..., 160:])[..., :active]
        return key, shared


class K8SwinDecoder(nn.Module):
    """Swin CSI decoder with explicit public G condition and zero-padded shared symbols."""

    def __init__(self, *, embed_dim=128, blocks=4, patch_size=4, window_size=4,
                 heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_size, self.window_size = patch_size, window_size
        self.height, self.width = N_SUBCARRIERS // patch_size, N_ANTENNAS // patch_size
        if self.height % window_size or self.width % window_size or embed_dim % heads:
            raise ValueError("Swin patch/window geometry or attention heads are incompatible")
        max_width = 4 + shared_dimension(1)
        self.fc = nn.Linear(2 * max_width + 3, self.height * self.width * embed_dim)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, heads, window_size, mlp_ratio) for _ in range(blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, 2 * patch_size**2)

    def tokens_from_symbols(self, key, shared, *, group_count):
        _groups(group_count)
        batch = key.shape[0] if torch.is_tensor(key) and key.ndim else 0
        active = shared_dimension(group_count)
        _complex(key, "key", (batch, K_USERS, 4))
        _complex(shared, "shared", (batch, K_USERS, active))
        if len({key.device, shared.device, self.fc.weight.device}) != 1:
            raise ValueError("Swin decoder inputs and parameters must share a device")
        padded = torch.zeros(batch, K_USERS, 160, dtype=shared.dtype, device=shared.device)
        padded[..., :active] = shared
        symbols = torch.cat((key, padded), dim=-1)
        real = torch.cat((symbols.real, symbols.imag), dim=-1)
        group_index = SUPPORTED_GROUP_COUNTS.index(group_count)
        condition = F.one_hot(torch.full((batch,), group_index, device=key.device), 3)
        condition = condition.to(dtype=real.dtype)[:, None].expand(-1, K_USERS, -1)
        codeword = torch.cat((real, condition), dim=-1).reshape(batch * K_USERS, -1)
        tokens = self.fc(codeword).view(batch * K_USERS, self.height * self.width, -1)
        for block in self.blocks:
            tokens = block(tokens, self.height, self.width)
        return tokens

    def tokens_to_channel(self, tokens, batch, *, normalize=True):
        if tuple(tokens.shape) != (batch * K_USERS, self.height * self.width, self.fc.out_features // (self.height * self.width)):
            raise ValueError("Swin decoder token geometry mismatch")
        features = self.norm(tokens) if normalize else tokens
        patches = self.final_proj(features)
        p = self.patch_size
        patches = patches.view(batch * K_USERS, self.height, self.width, 2, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        angular = patches.view(batch, K_USERS, 2, N_SUBCARRIERS, N_ANTENNAS)
        H_hat = inverse_angular_delay_transform(angular, N_SUBCARRIERS)
        if not bool(torch.isfinite(H_hat).all()):
            raise FloatingPointError("K8 Swin decoder produced nonfinite CSI")
        return H_hat

    def forward(self, key, shared, *, group_count):
        batch = key.shape[0] if torch.is_tensor(key) and key.ndim else 0
        tokens = self.tokens_from_symbols(key, shared, group_count=group_count)
        return self.tokens_to_channel(tokens, batch, normalize=True)


class K8MultiSourceFusionSwinReceiver(nn.Module):
    """Runnable multi-source conditioned Swin receiver prototype.

    The prototype adds convolutional features from received Y, estimated UL
    CSI, resource masks, calibration error, symbol reliability, N0 and public
    energy/group conditions to symbol-derived Swin tokens. It does not yet
    implement the planned cross-user self-attention, observation cross-
    attention, soft-codeword residual correction, or spatial cross-attention.
    """

    def __init__(self, *, embed_dim=128, blocks=2, patch_size=4, window_size=4,
                 heads=4, mlp_ratio=4.0, base_decoder=None):
        super().__init__()
        self.base_decoder = base_decoder or K8SwinDecoder(embed_dim=embed_dim, blocks=4,
            patch_size=patch_size, window_size=window_size, heads=heads,
            mlp_ratio=mlp_ratio)
        self.patch_size = patch_size
        self.height, self.width = self.base_decoder.height, self.base_decoder.width
        input_channels = 4 * N_ANTENNAS + 3 + 2 + 7
        self.side_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=7, padding=3),
            nn.GELU(), nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.joint_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, heads, window_size, mlp_ratio)
            for _ in range(blocks)
        ])
        self.joint_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _batch_scalar(value, batch, device, name, *, allow_zero=True):
        result = torch.as_tensor(value, dtype=torch.float32, device=device)
        if result.ndim == 0:
            result = result.expand(batch)
        if result.shape != (batch,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must be a finite scalar or [B]")
        if not bool((result >= 0 if allow_zero else result > 0).all()):
            raise ValueError(f"{name} is outside its valid range")
        return result

    def forward(self, key, shared, *, group_count, Y_rx, H_ul_est, layout,
                feedback_gamma_db, noise_variance, pilot_energy_per_ue,
                key_energy_per_ue, shared_energy_per_ue,
                channel_error_variance, key_reliability, shared_reliability):
        _groups(group_count)
        batch = key.shape[0] if torch.is_tensor(key) and key.ndim else 0
        active_shared = shared_dimension(group_count)
        _complex(key, "key soft symbols", (batch, K_USERS, 4))
        _complex(shared, "shared soft symbols", (batch, K_USERS, active_shared))
        _complex(Y_rx, "received feedback", (batch, N_ANTENNAS, N_SUBCARRIERS))
        _complex(H_ul_est, "estimated UL CSI", (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS))
        if len({key.device, shared.device, Y_rx.device, H_ul_est.device,
                self.base_decoder.fc.weight.device}) != 1:
            raise ValueError("Fusion receiver inputs and parameters must share a device")
        if type(layout) not in (_protocol.HybridLayout, _protocol.FDMALayout):
            raise TypeError("Fusion receiver requires an explicit HybridLayout or FDMALayout")
        if isinstance(layout, _protocol.HybridLayout) and layout.group_count != group_count:
            raise ValueError("Hybrid layout G does not match decoder group_count")
        if isinstance(layout, _protocol.FDMALayout) and group_count != 1:
            raise ValueError("FDMA control uses the G=1 decoder condition")

        key_rel = torch.as_tensor(key_reliability, dtype=torch.float32, device=key.device)
        shared_rel = torch.as_tensor(shared_reliability, dtype=torch.float32, device=key.device)
        if key_rel.shape != (batch, K_USERS, 4) or shared_rel.ndim != 3 or shared_rel.shape[:2] != (batch, K_USERS):
            raise ValueError("Soft-symbol reliability tensors have incompatible shapes")
        if not bool(torch.isfinite(key_rel).all() & torch.isfinite(shared_rel).all()
                    & (key_rel >= 0).all() & (key_rel <= 1).all()
                    & (shared_rel >= 0).all() & (shared_rel <= 1).all()):
            raise ValueError("Soft-symbol reliabilities must be finite and within [0,1]")

        pilot_mask = torch.zeros((K_USERS, N_SUBCARRIERS), dtype=torch.float32, device=key.device)
        key_mask = torch.zeros_like(pilot_mask)
        shared_mask = torch.zeros_like(pilot_mask)
        key_rel_map = torch.zeros((batch, K_USERS, N_SUBCARRIERS), dtype=torch.float32, device=key.device)
        shared_rel_map = torch.zeros_like(key_rel_map)
        for user, positions in enumerate(_protocol.PILOT_POSITIONS_BY_USER):
            pilot_mask[user, list(positions)] = 1.0
        if isinstance(layout, _protocol.HybridLayout):
            for user, positions in enumerate(layout.key_positions):
                key_mask[user, list(positions)] = 1.0
                key_rel_map[:, user, list(positions)] = key_rel[:, user]
            for members, positions in zip(layout.groups, layout.shared_positions_by_group):
                for user in members:
                    shared_mask[user, list(positions)] = 1.0
                    shared_rel_map[:, user, list(positions)] = shared_rel[:, user]
        else:
            if shared_rel.shape[-1] != layout.data_coordinates_per_user - 4:
                raise ValueError("FDMA path expects four packed key coordinates plus 20 shared coordinates")
            for user, positions in enumerate(layout.data_positions):
                shared_mask[user, list(positions)] = 1.0
                packed_rel = torch.cat((key_rel[:, user], shared_rel[:, user]), dim=-1)
                shared_rel_map[:, user, list(positions)] = packed_rel

        gamma = self._batch_scalar(feedback_gamma_db, batch, key.device, "feedback_gamma_db")
        n0 = self._batch_scalar(noise_variance, batch, key.device, "noise_variance", allow_zero=False)
        ep = self._batch_scalar(pilot_energy_per_ue, batch, key.device, "pilot_energy_per_ue", allow_zero=False)
        ek = self._batch_scalar(key_energy_per_ue, batch, key.device, "key_energy_per_ue")
        es = self._batch_scalar(shared_energy_per_ue, batch, key.device, "shared_energy_per_ue", allow_zero=False)
        cev = torch.as_tensor(channel_error_variance, dtype=torch.float32, device=key.device)
        if cev.ndim == 0:
            cev = cev.expand(batch, K_USERS)
        elif cev.shape == (K_USERS,):
            cev = cev[None].expand(batch, -1)
        if cev.shape != (batch, K_USERS) or not bool(torch.isfinite(cev).all() & (cev >= 0).all()):
            raise ValueError("channel_error_variance must be nonnegative [B,8]")
        ce_context = cev.mean(dim=1)
        context = torch.stack((gamma / 25.0,
            torch.log10(n0.clamp_min(1e-12)), ep / 64.0, ek / 64.0,
            es / 64.0, torch.log10(ce_context.clamp_min(1e-12)),
            torch.full_like(gamma, group_count / 4.0)), dim=1)
        context = context[:, None, :, None].expand(-1, K_USERS, -1, N_SUBCARRIERS)

        y_users = Y_rx[:, None].expand(-1, K_USERS, -1, -1)
        y_features = torch.cat((y_users.real, y_users.imag), dim=2)
        ul_features = torch.cat((H_ul_est.real, H_ul_est.imag), dim=2)
        resources = torch.stack((pilot_mask, key_mask, shared_mask), dim=1)[None]
        resources = resources.expand(batch, -1, -1, -1)
        reliability = torch.stack((key_rel_map, shared_rel_map), dim=2)
        features = torch.cat((y_features, ul_features, resources, reliability, context), dim=2)
        features = features.reshape(batch * K_USERS, features.shape[2], N_SUBCARRIERS)

        latent = self.base_decoder.tokens_from_symbols(key, shared, group_count=group_count)
        side = torch.nn.functional.avg_pool1d(self.side_encoder(features),
            kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        side = side[:, :, None, :].expand(-1, -1, self.width, -1).reshape_as(latent)
        tokens = latent + side
        for block in self.joint_blocks:
            tokens = block(tokens, self.height, self.width)
        return self.base_decoder.tokens_to_channel(tokens, batch, normalize=True)


# Backward-compatible symbol name. The implementation remains the multi-source
# fusion prototype above; it is not the planned full Joint Transformer.
K8JointFeedbackDecoder = K8MultiSourceFusionSwinReceiver


class K8SwinCSIModel(nn.Module):
    """Separate encoder/decoder calls for UE encoding and BS reconstruction."""

    def __init__(self, **swin_kwargs):
        super().__init__()
        self.encoder = K8SwinEncoder(**swin_kwargs)
        self.decoder = K8SwinDecoder(**swin_kwargs)
        self.joint_decoder = K8MultiSourceFusionSwinReceiver(base_decoder=self.decoder,
            embed_dim=swin_kwargs.get("embed_dim", 128),
            patch_size=swin_kwargs.get("patch_size", 4),
            window_size=swin_kwargs.get("window_size", 4),
            heads=swin_kwargs.get("heads", 4),
            mlp_ratio=swin_kwargs.get("mlp_ratio", 4.0))

    def encode(self, H_dl, *, group_count):
        return self.encoder(H_dl, group_count=group_count)

    def decode(self, key, shared, *, group_count):
        return self.decoder(key, shared, group_count=group_count)

    def encode_fdma(self, H_dl):
        """Pack the first 24 complex encoder coordinates for FDMA data REs."""
        key, shared = self.encode(H_dl, group_count=1)
        return torch.cat((key, shared), dim=-1)[..., :24]

    def decode_feedback(self, key, shared, **receiver_inputs):
        return self.joint_decoder(key, shared, **receiver_inputs)

    def decode_fdma_feedback(self, data_symbols_hat, **receiver_inputs):
        if (data_symbols_hat.dtype != torch.complex64 or data_symbols_hat.ndim != 3
                or tuple(data_symbols_hat.shape[1:]) != (K_USERS, 24)):
            raise ValueError("FDMA recovered code must be complex64 [B,8,24]")
        key = data_symbols_hat[..., :4]
        shared = torch.zeros((data_symbols_hat.shape[0], K_USERS, shared_dimension(1)),
            dtype=data_symbols_hat.dtype, device=data_symbols_hat.device)
        shared[..., :20] = data_symbols_hat[..., 4:]
        reliability = receiver_inputs.pop("reliability")
        return self.joint_decoder(key, shared, group_count=1,
            key_reliability=reliability[..., :4],
            shared_reliability=reliability[..., 4:], **receiver_inputs)
