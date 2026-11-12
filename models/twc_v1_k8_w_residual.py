"""Independent per-tone residual-W parameterization for the K8 rate control.

The original CSI decoder and fixed-alpha RZF implementation are unchanged.
Only estimated CSI, public feedback gamma and fixed resource/user coordinates
enter the added head. True DL channels are accepted solely by the separate
training/scoring objective, never by either decoder forward.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn


METHODS = ("rzf_continue", "w_residual")
K, NT, NSC = 8, 32, 256
ALPHA = .03
HEAD_WIDTH, HEAD_LAYERS, HEAD_HEADS = 64, 2, 4
RESIDUAL_SCALE = 1. / math.sqrt(NT * K)
DEFAULT_HEAD_SEED = 20261112


def _rate_module():
    """Read the maintained math helper without importing side-effectful utils."""
    name = "_twc_v1_k8_w_residual_rate"
    path = Path(__file__).resolve().parents[1] / "utils/twc_v1_k8_icc_rate.py"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("Residual-W rate module identity collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate the unchanged rate helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _channel(value):
    if (not torch.is_tensor(value) or value.dtype != torch.complex64
            or value.ndim != 4 or value.shape[0] < 1
            or tuple(value.shape[1:]) != (K, NT, NSC)):
        raise ValueError("Estimated DL CSI must be complex64 [B,8,32,256]")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Estimated DL CSI is nonfinite")
    return value


def _public_gamma(value, batch, device):
    try:
        original = torch.as_tensor(value)
        if (original.is_complex() or original.dtype == torch.bool
                or original.requires_grad):
            raise ValueError("Public gamma must be real, nonlearned and nonboolean")
        gamma = original.to(device=device, dtype=torch.float32)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Public gamma must be a scalar or [B] tensor") from error
    if gamma.ndim == 0:
        gamma = gamma.expand(batch)
    if (gamma.shape != (batch,) or not bool(torch.isfinite(gamma).all()
            & (gamma >= 0.).all() & (gamma <= 25.).all())):
        raise ValueError("Public gamma must be a scalar or [B] in [0,25] dB")
    return gamma


def _precoder(value):
    if (not torch.is_tensor(value) or value.dtype != torch.complex64
            or value.ndim != 4 or value.shape[0] < 1
            or tuple(value.shape[1:]) != (NSC, NT, K)):
        raise ValueError("Precoder must be complex64 [B,256,32,8]")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Precoder is nonfinite")
    return value


def normalize_precoder(value):
    """Project total matrix power to one per tone; never normalize columns."""
    value = _precoder(value)
    power = value.abs().square().sum(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(power).all() & (power > 1e-20).all()):
        raise FloatingPointError("Cannot normalize a zero or nonfinite precoder")
    normalized = value / power.sqrt()
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("Normalized precoder is nonfinite")
    return normalized


class ResidualWHead(nn.Module):
    """Two all-eight-user attention layers, independently at each tone."""
    def __init__(self):
        super().__init__()
        # 32 complex estimated channel coefficients, gamma, UID and tone.
        self.input_projection = nn.Linear(2 * NT + 1 + K + 3, HEAD_WIDTH)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(HEAD_WIDTH, HEAD_HEADS,
                dim_feedforward=2 * HEAD_WIDTH, dropout=0., activation="gelu",
                batch_first=True, norm_first=True)
            for _ in range(HEAD_LAYERS)
        ])
        self.output_norm = nn.LayerNorm(HEAD_WIDTH)
        self.final_projection = nn.Linear(HEAD_WIDTH, 2 * NT)
        frequency = torch.linspace(-1., 1., NSC)
        positions = torch.stack((frequency, torch.sin(math.pi * frequency),
                                 torch.cos(math.pi * frequency)), dim=-1)
        self.register_buffer("fixed_user_identity", torch.eye(K), persistent=False)
        self.register_buffer("fixed_frequency", positions, persistent=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        # Hidden attention is initialized normally; only the last map is zero.
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, H_hat, *, gamma_db):
        H_hat = _channel(H_hat)
        batch = H_hat.shape[0]
        if (self.input_projection.weight.device != H_hat.device
                or self.input_projection.weight.dtype != torch.float32):
            raise ValueError("Residual head must be FP32 on the estimated CSI device")
        gamma = _public_gamma(gamma_db, batch, H_hat.device)
        channel = torch.view_as_real(H_hat.permute(0, 3, 1, 2).contiguous()).flatten(-2)
        public_gamma = ((gamma - 12.5) / 12.5)[:, None, None, None].expand(batch, NSC, K, 1)
        users = self.fixed_user_identity[None, None].expand(batch, NSC, K, K)
        frequency = self.fixed_frequency[None, :, None].expand(batch, NSC, K, 3)
        features = torch.cat((channel, public_gamma, users, frequency), dim=-1)
        tokens = self.input_projection(features).reshape(batch * NSC, K, HEAD_WIDTH)
        for layer in self.layers:
            tokens = layer(tokens)
        output = self.final_projection(self.output_norm(tokens))
        output = output.reshape(batch, NSC, K, NT, 2).contiguous()
        delta = torch.view_as_complex(output).permute(0, 1, 3, 2).contiguous()
        delta = delta * RESIDUAL_SCALE
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Residual-W head produced nonfinite values")
        return delta


class PrecodingDecoder(nn.Module):
    """Trainable original decoder plus optional zero-initialized residual W."""
    def __init__(self, base_decoder, method, *, head_seed=DEFAULT_HEAD_SEED):
        super().__init__()
        if not isinstance(base_decoder, nn.Module):
            raise ValueError("An explicit original decoder module is required")
        if method not in METHODS:
            raise ValueError("Only rzf_continue and w_residual are supported")
        if type(head_seed) is not int or not 0 <= head_seed <= 2**32 - 1:
            raise ValueError("Explicit uint32 residual-head seed required")
        self.base_decoder = base_decoder
        self.method = method
        self.head_seed = head_seed
        self.arm = getattr(base_decoder, "arm", None)
        if method == "w_residual":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(head_seed)
                self.residual_head = ResidualWHead()
        else:
            self.residual_head = None

    def forward(self, Y_rx, H_ul_est, soft, reliability, pilots, *, gamma_db,
                layout=None, prior=None, ce_error=None):
        # Explicitly retain the original public receiver interface.
        result = self.base_decoder(Y_rx=Y_rx, H_ul_est=H_ul_est, soft=soft,
            reliability=reliability, pilots=pilots, gamma_db=gamma_db,
            layout=layout, prior=prior, ce_error=ce_error)
        if not isinstance(result, dict) or "H_hat" not in result:
            raise ValueError("Original decoder must return a dict containing H_hat")
        if any(key in result for key in ("W", "W_base", "delta_W")):
            raise ValueError("Original decoder already contains precoding output keys")
        H_hat = _channel(result["H_hat"])
        gamma = _public_gamma(gamma_db, H_hat.shape[0], H_hat.device)
        W_base, _ = _rate_module()._downlink().design_rzf(H_hat, ALPHA)
        if self.residual_head is None:
            delta = torch.zeros_like(W_base)
            W = W_base
        else:
            delta = self.residual_head(H_hat, gamma_db=gamma)
            W = normalize_precoder(W_base + delta)
        return dict(result, W=W, W_base=W_base, delta_W=delta)

    def head_parameter_count(self):
        return 0 if self.residual_head is None else sum(p.numel() for p in self.residual_head.parameters())

    def metadata(self):
        base_metadata = getattr(self.base_decoder, "metadata", None)
        return dict(architecture="twc_v1_k8_w_residual_v1", method=self.method,
            arm=self.arm, alpha=ALPHA, residual_scale=RESIDUAL_SCALE,
            head_seed=self.head_seed, head_width=HEAD_WIDTH,
            head_layers=HEAD_LAYERS if self.residual_head is not None else 0,
            head_attention_heads=HEAD_HEADS, head_dropout=0.,
            head_inputs=["H_hat", "public_gamma", "fixed_user_identity", "fixed_frequency"],
            precoder_shape=[NSC, NT, K], normalization="per-tone total Frobenius power=1",
            base_decoder_parameters=sum(p.numel() for p in self.base_decoder.parameters()),
            head_parameters=self.head_parameter_count(),
            total_parameters=sum(p.numel() for p in self.parameters()),
            base_decoder=base_metadata() if callable(base_metadata) else None,
            true_channel_forward_input=False, rzf_output_space_expansion_claim=False)


def precoding_objective(W, H_true, *, downlink_snr_db=25.):
    """Negative mean true-channel rate; labels are loss-only, not head inputs."""
    W, H_true = _precoder(W), _channel(H_true)
    if H_true.shape[0] != W.shape[0] or H_true.device != W.device:
        raise ValueError("Matched precoder and true DL labels required")
    power = W.abs().square().sum(dim=(-2, -1))
    if not bool(((power - 1.).abs() <= 1e-5).all()):
        raise FloatingPointError("Precoder violates per-tone total power=1")
    metrics = _rate_module().differentiable_rates(H_true.permute(0, 3, 1, 2), W,
                                                downlink_snr_db=downlink_snr_db)
    loss = -metrics["sum_rate_bps_hz"].mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Residual-W rate objective is nonfinite")
    return loss, metrics
