"""Fixed grouped-feedback layouts and addressable randomness.

The module itself is standard-library only. Tensor functions import Torch lazily.
Load this real file directly for stdlib dry-runs: the historical utils package
initializer has unrelated GPU/Sionna side effects and is intentionally untouched.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import random

ARMS = ("legacy_all4", "ordered_all4", "legacy_group2", "ordered_group2", "ordered_fdma")
PROTOCOL = "grouped_feedback_energy64_absolute_noise_v1"
ARCHITECTURE = "grouped_codec_v1"
NORM_ENERGY_FLOOR = 1e-12
ENERGY_ATOL = 1e-4


@dataclass(frozen=True)
class LayoutSpec:
    layout: str

    def __post_init__(self):
        if self.layout not in ("all4", "group2", "fdma"):
            raise ValueError("Only fixed all4, group2 and fdma layouts are supported.")
        occupancy = [sum(q in support for support in self.supports) for q in range(256)]
        if set(occupancy) != {self.users_per_resource}:
            raise ValueError("Invalid physical support occupancy.")

    @property
    def active_per_user(self):
        return {"all4": 256, "group2": 128, "fdma": 64}[self.layout]

    @property
    def users_per_resource(self):
        return {"all4": 4, "group2": 2, "fdma": 1}[self.layout]

    @property
    def occupied_resources(self):
        return 256

    @property
    def active_slots(self):
        return tuple(sb * 64 + j for sb in range(4)
                     for j in range(self.active_per_user // 4))

    @property
    def supports(self):
        if self.layout == "all4":
            return (tuple(range(256)),) * 4
        if self.layout == "group2":
            return tuple(tuple(q for q in range(256) if q % 4 // 2 == k // 2)
                         for k in range(4))
        return tuple(tuple(range(k, 256, 4)) for k in range(4))

    @property
    def groups(self):
        return {"all4": ((0, 1, 2, 3),), "group2": ((0, 1), (2, 3)),
                "fdma": ((0,), (1,), (2,), (3,))}[self.layout]

    @property
    def canonical_to_physical(self):
        values = []
        for support in self.supports:
            mapping = [-1] * 256
            for slot, physical in zip(self.active_slots, support):
                mapping[slot] = physical
            values.append(tuple(mapping))
        return tuple(values)

    def to_dict(self):
        return {
            "layout": self.layout, "users": 4, "antennas": 32, "subcarriers": 256,
            "occupied_resources": 256, "canonical_slots_per_user": 256,
            "active_per_user": self.active_per_user, "users_per_resource": self.users_per_resource,
            "groups": [list(group) for group in self.groups],
            "active_slots": list(self.active_slots),
            "physical_supports": [list(support) for support in self.supports],
            "canonical_to_physical": [list(row) for row in self.canonical_to_physical],
            "slot_order": "subband*64+j; each subband prefix; ascending physical support",
        }


def layout_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown grouped-codec arm.")
    return LayoutSpec(arm.split("_", 1)[1])


@dataclass(frozen=True)
class RandomAddress:
    seed: int
    split: str
    index: int
    stream: str

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0 or type(self.index) is not int or self.index < 0:
            raise ValueError("Random address seed/index must be nonnegative integers.")
        if self.split not in ("init", "train", "validation", "test", "preflight"):
            raise ValueError("Unknown random-address namespace.")
        if self.stream not in ("gamma", "standard_noise", "ue_legacy", "ue_ordered", "bs"):
            raise ValueError("Unknown random stream.")

    def to_dict(self):
        return {"schema": PROTOCOL, "seed": self.seed, "split": self.split,
                "index": self.index, "stream": self.stream}

    @property
    def seed_value(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def sample_gamma(seed, global_step):
    return random.Random(RandomAddress(seed, "train", global_step, "gamma").seed_value).uniform(0.0, 25.0)


def standard_noise(seed, split, index, batch_size, device="cpu"):
    """Complex eps with independent standard-normal real/imag, not CN(0,1).

    A fresh private CPU generator makes the complete physical tensor independent
    of model dropout, constructors, device RNG and all other calls. It is then
    transferred without resampling. Gamma and arm never enter this address.
    """
    import torch
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Noise generation requires a positive complete-frame batch size.")
    address = RandomAddress(seed, split, index, "standard_noise")
    generator = torch.Generator(device="cpu").manual_seed(address.seed_value)
    parts = torch.randn((batch_size, 32, 256, 2), generator=generator, dtype=torch.float32)
    return torch.view_as_complex(parts).to(device=device)


def noise_variance(gamma_db):
    if isinstance(gamma_db, bool) or not isinstance(gamma_db, (int, float)):
        raise TypeError("Public gamma_db must be a nonlearned real scalar.")
    gamma = float(gamma_db)
    if not math.isfinite(gamma) or not 0 <= gamma <= 25:
        raise ValueError("Public gamma_db must be finite and within [0,25].")
    return (64.0 / 256.0) * 10.0 ** (-gamma / 10.0)


def protocol_metadata():
    return {
        "schema": PROTOCOL, "energy_per_frame_per_ue": 64.0,
        "gamma_definition": "10*log10(64/(256*N0)); block-average transmit reference, not received SNR",
        "N0_definition": "(64/256)*10**(-gamma_db/10)", "gamma_range_db": [0, 25],
        "physical_noise": "sqrt(N0/2)*(eps_real+1j*eps_imag); full [B,32,256] once",
        "standard_noise_components": "independent N(0,1), private CPU Torch generator then device transfer",
        "normalization": "8*u_active/sqrt(sum(abs(u_active)**2)); no epsilon",
        "norm_energy_fail_floor": NORM_ENERGY_FLOOR, "energy_check_atol": ENERGY_ATOL,
        "receiver": "actual active 1/2/4-user solve, complex64",
        "receiver_ridge": "N0/(64/D_active)+1e-6",
        "reliability": "clip(1-real(diag(lambda_rx*solve(A,I))),0,1); nominal-power proxy, not calibrated precision",
        "csi_mode": "perfect UL CSI", "pilots": False, "keys": False,
        "power_ema": False, "learned_power_control": False,
        "random_address": "SHA256(canonical JSON schema/seed/split/index/stream), first64bits masked to63",
        "gamma_stream": "stdlib private Random; train global_step; independent of standard_noise",
        "noise_stream": "no arm/encoder/gamma component; shared complete tensor across methods",
        "validation_test_namespaces": ["validation", "test"], "historical_protocol_compatible": False,
    }


def tensor_geometry(layout, device="cpu"):
    import torch
    if not isinstance(layout, LayoutSpec):
        raise TypeError("Explicit LayoutSpec required.")
    support = torch.tensor(layout.canonical_to_physical, dtype=torch.long, device=device)
    physical = torch.zeros((4, 256), dtype=torch.bool, device=device)
    physical.scatter_(1, torch.tensor(layout.supports, device=device), True)
    return {
        "support_sc": support, "active_mask": support >= 0, "physical_mask": physical,
        "active_slots": torch.tensor(layout.active_slots, dtype=torch.long, device=device),
        "physical_supports": torch.tensor(layout.supports, dtype=torch.long, device=device),
        "re_positions": torch.arange(256, device=device),
    }


def _complex_tensor(value, name, shape=None):
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype != torch.complex64:
        raise TypeError(name + " must be a genuine complex64 tensor.")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(name + " shape mismatch: " + str(tuple(value.shape)))
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(name + " contains nonfinite values; no discarded/replaced frame.")


def _public_noise(N0):
    if isinstance(N0, bool) or not isinstance(N0, (int, float)):
        raise TypeError("Public N0 must be a nonlearned scalar protocol parameter.")
    if not math.isfinite(N0) or not noise_variance(25) <= N0 <= noise_variance(0):
        raise ValueError("Public N0 lies outside the fixed gamma[0,25] domain.")
    return float(N0)


def normalize_and_map(raw, layout):
    """Return X[B,4,256], canonical normalized truth, and actual energy[B,4]."""
    import torch
    if raw.ndim != 3 or raw.shape[0] < 1:
        raise ValueError("Expected nonempty canonical raw codewords.")
    _complex_tensor(raw, "Raw codeword", (raw.shape[0], 4, 256))
    geometry = tensor_geometry(layout, raw.device)
    active = raw.index_select(-1, geometry["active_slots"])
    energy = active.abs().square().sum(-1, keepdim=True)
    if not bool(torch.isfinite(energy).all() & (energy > NORM_ENERGY_FLOOR).all()):
        bad = (~torch.isfinite(energy) | (energy <= NORM_ENERGY_FLOOR)).nonzero().detach().cpu().tolist()
        raise FloatingPointError("Near-zero/nonfinite active energy; fail-stop evidence frame/UE/axis=" + repr(bad[:16]))
    normalized = 8.0 * active / energy.sqrt()
    shape = (raw.shape[0], 4, 256)
    canonical = raw.new_zeros(shape).scatter(
        -1, geometry["active_slots"][None, None].expand(raw.shape[0], 4, -1), normalized)
    physical = raw.new_zeros(shape).scatter(
        -1, geometry["physical_supports"][None].expand(raw.shape[0], -1, -1), normalized)
    actual = physical.abs().square().sum(-1)
    if not bool(torch.isfinite(actual).all() & ((actual - 64.0).abs() <= ENERGY_ATOL).all()):
        raise FloatingPointError("Exact per-frame/UE energy normalization failed FP32 verification.")
    return physical, canonical, actual


def transmit(H_ul, X, layout, *, N0, standard_noise):
    """One four-UE full-grid MAC, followed by exactly one absolute-noise addition."""
    import torch
    noise = _public_noise(N0)
    if H_ul.ndim != 4 or H_ul.shape[0] < 1:
        raise ValueError("Expected nonempty physical UL channel.")
    batch = H_ul.shape[0]
    _complex_tensor(H_ul, "Physical UL", (batch, 4, 32, 256))
    _complex_tensor(X, "Physical TX", (batch, 4, 256))
    _complex_tensor(standard_noise, "Standard physical noise", (batch, 32, 256))
    if standard_noise.requires_grad:
        raise ValueError("Standard physical noise must be nonlearned.")
    if H_ul.device != X.device or standard_noise.device != X.device:
        raise ValueError("Physical tensors must share a device.")
    geometry = tensor_geometry(layout, X.device)
    if bool((X[:, ~geometry["physical_mask"]] != 0).any()):
        raise ValueError("Nonzero transmission on an inactive physical resource.")
    Y = (H_ul * X[:, :, None, :]).sum(dim=1)
    return Y + math.sqrt(noise / 2.0) * standard_noise


def recover(Y, H_ul_est, layout, *, N0):
    """Public receiver; no sent codeword, DL target, noise sample or clean Y."""
    import torch
    noise = _public_noise(N0)
    if Y.ndim != 3 or Y.shape[0] < 1:
        raise ValueError("Expected nonempty physical observations.")
    batch = Y.shape[0]
    _complex_tensor(Y, "Received Y", (batch, 32, 256))
    _complex_tensor(H_ul_est, "Receiver UL estimate", (batch, 4, 32, 256))
    if Y.device != H_ul_est.device:
        raise ValueError("Receiver tensors must share a device.")
    geometry = tensor_geometry(layout, Y.device)
    nominal_power = 64.0 / layout.active_per_user
    ridge = noise / nominal_power + 1e-6
    estimates, reliabilities = [None] * 4, [None] * 4
    for members in layout.groups:
        users = torch.tensor(members, device=Y.device)
        positions = torch.tensor(layout.supports[members[0]], device=Y.device)
        g = H_ul_est.index_select(1, users).index_select(-1, positions).permute(0, 3, 2, 1)
        y = Y.index_select(-1, positions).transpose(1, 2).unsqueeze(-1)
        gh = g.conj().transpose(-2, -1)
        identity = torch.eye(len(members), device=Y.device, dtype=torch.complex64)
        system = gh @ g + ridge * identity
        z = torch.linalg.solve(system, gh @ y).squeeze(-1)
        inv = torch.linalg.solve(system, identity.expand(batch, len(positions), -1, -1))
        reliability = (1.0 - (ridge * inv).diagonal(dim1=-2, dim2=-1).real).clamp(0.0, 1.0)
        for local, user in enumerate(members):
            estimates[user] = z[:, :, local]
            reliabilities[user] = reliability[:, :, local]
    indices = geometry["active_slots"][None, None].expand(batch, 4, -1)
    z_hat = Y.new_zeros((batch, 4, 256)).scatter(-1, indices, torch.stack(estimates, dim=1))
    reliability = Y.real.new_zeros((batch, 4, 256)).scatter(-1, indices, torch.stack(reliabilities, dim=1))
    if not bool(torch.isfinite(z_hat).all() & torch.isfinite(reliability).all()):
        raise FloatingPointError("Nonfinite public receiver result; no repair/fallback.")
    return {"z_hat": z_hat, "reliability": reliability,
            "lambda_rx": Y.real.new_tensor(ridge), "nominal_coordinate_power": nominal_power}


BS_SETTINGS = {
    "width": 128, "heads": 4, "joint_layers": 2, "joint_ffn": 256,
    "joint_activation": "GELU", "pre_layer_norm": True, "dropout": 0.0,
    "memory_tokens": 256, "canonical_queries": 1024, "spatial_queries": 2048,
    "query_features": 15, "physical_resource_bias": "true canonical_to_physical q, never slot index",
    "memory_features": ["noisy_Y", "perfect_UL", "physical_q_coordinates",
                        "public_active_user_mask", "log1p_reliability_proxy", "gamma_db/25"],
    "inactive_policy": "zero initial query and every residual update; mask self/spatial context keys; mask corrected FC input",
    "codeword_residual_weight_std": 0.001, "spatial_cross_output_weight_std": 0.001,
    "other_extra_linear_std": 0.02, "head_bias_init": 0.02,
    "codeword_gate": False, "receive_power_normalization": False,
    "fc_shape": [512, 65536], "codeword_real_order": "all real then all imaginary",
    "patch_grid": [64, 8], "patch_size": 4, "full_delay_taps": 256, "antennas": 32,
    "swin_blocks": 4, "window_size": 4, "shifted_windows": False, "swin_mlp_ratio": 4.0,
    "implementation": "existing 03 Swin blocks; new explicitly mapped/masked global Joint decoder",
    "query_chunk": 64, "all_keys_per_chunk": True,
    "activation_checkpoint": True, "checkpoint_use_reentrant": False,
    "rzf_regularization": 1e-3, "precoder_power_per_subcarrier": 1.0,
    "public_reliability": "nominal coordinate-power proxy; not posterior precision",
}


def architecture_metadata(arm):
    from copy import deepcopy
    layout = layout_for_arm(arm)
    legacy = arm.startswith("legacy_")
    return {
        "architecture": ARCHITECTURE, "model_class": "GroupedCodecPrecoder", "arm": arm,
        "ue_family": "legacy" if legacy else "ordered", "layout_spec": layout.to_dict(),
        "protocol": protocol_metadata(), "bs": deepcopy(BS_SETTINGS),
        "ue": ({
            "implementation": "unchanged AdaptiveHybridEncoder AS forward plus ConditionEmbed",
            "stored_original_fdma_branch": True, "fdma_branch_executed": False,
            "condition": "[0,D_active/256,gamma_db/25]; active length is not physical feedback budget",
            "direction_normalization": True, "stride": 4, "width": 256, "heads": 8,
            "layers": 4, "ffn": 2048, "dropout": 0.1,
            "serialization": "subband-major; each head all64 real then all64 imag",
        } if legacy else {
            "implementation": "full angular-delay ordered-window encoder",
            "input": [2, 256, 32], "patch_size": 4, "width": 128, "heads": 4,
            "blocks": 4, "shifted_windows": False, "window_size": 4, "mlp_ratio": 4.0,
            "ordered_pool": [16, 2, 128], "pooling": "mean within each 4x4 token window only",
            "fc_shape": [4096, 512], "serialization": "all256 real then all256 imag; canonical slot=subband*64+j",
        }),
        "initialization": "new random UE and BS; independent addressed CPU RNG streams; no historical weights",
        "ue_frozen": False, "power_statistics_buffer": False,
    }
