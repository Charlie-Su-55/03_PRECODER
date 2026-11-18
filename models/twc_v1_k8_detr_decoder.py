"""DETR-style direct CSI decoder baseline for the K8 frame.

Adapted (not reproduced) from the Round-8 DirectDETRDecoder of the 08 project
(icc08/resource_research/round8/decoder.py): five pre-norm transformer decoder
layers of width 512, 8 heads, FFN 2048, no dropout. There are K x 32 angular
queries, one per (user, BS angle), built from the estimated UL angular vector,
the LMMSE soft codeword, the public resource map, user identity, reliability
and the feedback SNR. The memory has 32 angular tokens of the received frame
plus the pilots. Each query reads out all 256 frequency coefficients of its
angle directly; there is no codeword-residual head, FC expansion or Swin
decoder. Same call signature and output as the 03 joint decoder, so it plugs
into the same encoder, receiver, RZF and objective.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.twc_v1_k8_icc_aligned import support_geometry, K, NT, NSC, DMAX

WIDTH, LAYERS, HEADS, FFN = 512, 5, 8, 2048


def _coordinates(index, size):
    angle = 2*torch.pi*index.float()/size
    return torch.stack((torch.cos(angle), torch.sin(angle), index.float()/size), -1)


class K8DETRDecoder(nn.Module):
    def __init__(self, layout, *, activation_checkpointing=True):
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.memory_Y = nn.Linear(2*NSC, WIDTH)
        self.memory_pilot = nn.Linear(2*K*NSC, WIDTH)
        self.memory_position = nn.Linear(3, WIDTH)
        self.memory_reliability = nn.Linear(K, WIDTH)
        self.memory_noise = nn.Linear(1, WIDTH)
        self.memory_norm = nn.LayerNorm(WIDTH)
        self.query_UL = nn.Linear(2*NSC, WIDTH)
        self.query_soft_codeword = nn.Linear(2*DMAX, WIDTH)
        self.query_resource = nn.Linear(5*DMAX, WIDTH)
        self.query_position = nn.Linear(3, WIDTH)
        self.query_user = nn.Linear(K, WIDTH)
        self.query_exclusive_summary = nn.Linear(4, WIDTH)
        self.query_reliability = nn.Linear(4, WIDTH)
        self.query_noise = nn.Linear(1, WIDTH)
        self.query_norm = nn.LayerNorm(WIDTH)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=WIDTH, nhead=HEADS, dim_feedforward=FFN,
            dropout=0., activation='gelu', batch_first=True, norm_first=True) for _ in range(LAYERS)])
        self.output_norm = nn.LayerNorm(WIDTH)
        self.physical_readout = nn.Linear(WIDTH, 2*NSC)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02); nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.physical_readout.weight, std=.001)
        support, active, exclusive = support_geometry(layout)
        resource = torch.cat((_coordinates(support.clamp_min(0), NSC),
            torch.stack((exclusive, active & ~exclusive), -1).float()), -1)*active[..., None]
        self.register_buffer('_fixed_active', active, persistent=False)
        self.register_buffer('_fixed_exclusive', exclusive, persistent=False)
        self.register_buffer('_fixed_resource', resource.reshape(K, 5*DMAX), persistent=False)
        self.register_buffer('_fixed_angle', _coordinates(torch.arange(NT), NT), persistent=False)
        self.register_buffer('_fixed_user', torch.eye(K), persistent=False)

    def _run(self, layer, query, memory):
        if self.activation_checkpointing and self.training:
            return checkpoint(layer, query, memory, use_reentrant=False)
        return layer(query, memory)

    def forward(self, Y_rx, H_ul_est, soft, reliability, pilots, *, gamma_db, layout=None, prior=None, ce_error=None):
        batch = Y_rx.shape[0]
        if (Y_rx.shape != (batch, NT, NSC) or H_ul_est.shape != (batch, K, NT, NSC)
                or soft.shape != (batch, K, DMAX) or reliability.shape != (batch, K, DMAX)
                or pilots.shape != (batch, K, NSC)):
            raise ValueError('K8 DETR decoder input shapes')
        gamma = torch.as_tensor(gamma_db, dtype=torch.float32, device=Y_rx.device).reshape(-1).expand(batch)
        feature = ((gamma-12.5)/12.5)[:, None]
        active, exclusive = self._fixed_active, self._fixed_exclusive
        soft = soft*active[None]
        precision = torch.log1p(reliability.clamp_min(0))*active[None]
        count = active.sum(-1).clamp_min(1)
        precision_mean = precision.sum(-1)/count                                   # [B,K]
        angular_Y = torch.fft.fft(Y_rx, dim=1, norm='ortho')                        # [B,32,256]
        memory = self.memory_norm(self.memory_Y(torch.view_as_real(angular_Y.contiguous()).flatten(-2))
            + self.memory_pilot(torch.view_as_real(pilots.contiguous()).flatten(1))[:, None]
            + self.memory_position(self._fixed_angle)[None]
            + self.memory_reliability(precision_mean)[:, None]
            + self.memory_noise(feature)[:, None])
        angular_H = torch.fft.fft(H_ul_est, dim=2, norm='ortho')                    # [B,K,32,256]
        channel = torch.view_as_real(angular_H.contiguous()).flatten(-2)
        soft_real = torch.view_as_real(soft.contiguous()).flatten(-2)
        ex_count = exclusive.sum(-1).clamp_min(1)
        ex_mean = (soft*exclusive).sum(-1)/ex_count
        summary = torch.stack((ex_mean.real, ex_mean.imag, (soft.abs().square()*exclusive).sum(-1)/ex_count,
            (soft.abs()*exclusive).sum(-1)/ex_count), -1)
        rel = torch.stack((precision_mean, precision.masked_fill(~active[None], torch.inf).amin(-1),
            precision.amax(-1), (precision*exclusive).sum(-1)/ex_count), -1)
        query = (self.query_UL(channel) + self.query_soft_codeword(soft_real)[:, :, None]
            + self.query_resource(self._fixed_resource)[None, :, None]
            + self.query_position(self._fixed_angle)[None, None]
            + self.query_user(self._fixed_user)[None, :, None]
            + self.query_exclusive_summary(summary)[:, :, None]
            + self.query_reliability(rel)[:, :, None] + self.query_noise(feature)[:, None, None])
        query = self.query_norm(query.reshape(batch, K*NT, WIDTH))
        for layer in self.layers:
            query = self._run(layer, query, memory)
        coeff = self.physical_readout(self.output_norm(query)).reshape(batch, K, NT, NSC, 2)
        H_hat = torch.fft.ifft(torch.view_as_complex(coeff.float().contiguous()), dim=2, norm='ortho')
        if not bool(torch.isfinite(H_hat).all()):
            raise FloatingPointError('DETR decoder returned nonfinite CSI')
        return dict(H_hat=H_hat)

    def metadata(self):
        return dict(architecture='k8_detr_direct_angular_frequency_v1', width=WIDTH, layers=LAYERS, heads=HEADS,
            feedforward=FFN, queries=K*NT, memory_tokens=NT, parameters=sum(p.numel() for p in self.parameters()),
            adapted_from='08 round8 DirectDETRDecoder', exact_reproduction=False)
