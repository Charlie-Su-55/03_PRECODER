#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TWC-compatible TUB-style no-prior VQ-SSCC baseline.

Pipeline:
    CSI -> AD transform -> Encoder -> SAMM top-k
        -> VQ -> ADM completion -> Decoder -> reconstructed CSI -> common RZF

Training:
    Stage 1: Encoder/Decoder autoencoder pretraining
    Stage 2: Frozen Encoder/Decoder + random token selection + VQ/ADM
    Stage 3: SAMM top-k + end-to-end reconstruction fine-tuning

Digital source budget:
    4x4 latent grid -> 16 token positions
    J=256 -> 8 bits/VQ index
    keep_k=6
    position bits = ceil(log2(C(16,6))) = 13
    index bits = 6*8 = 48
    total source bits = 61 bits/UE

Later with Rc=2/3 and 64-QAM:
    61 information bits -> padded coded block -> 16 complex uses/UE
    K=8 -> 128 total complex channel uses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from models.baseline_common import angular_delay_transform, closed_form_rzf, inverse_angular_delay_transform
from utils import setup_gpu


# =============================================================================
# Utilities
# =============================================================================

def group_norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def nmse_ratio(x_hat, x):
    err = (x_hat - x).square().sum(dim=(1, 2, 3))
    ref = x.square().sum(dim=(1, 2, 3)).clamp_min(1e-10)
    return err / ref


def mean_se(x):
    x = torch.cat(x).float()
    return x.mean().item(), (x.std(unbiased=True) / math.sqrt(x.numel())).item()


def combinatorial_source_bits(num_tokens, keep_k, num_embeddings):
    pos_bits = 0 if keep_k in (0, num_tokens) else math.ceil(math.log2(math.comb(num_tokens, keep_k)))
    index_bits = keep_k * int(math.log2(num_embeddings))
    return pos_bits, index_bits, pos_bits + index_bits


# =============================================================================
# Encoder / Decoder
# =============================================================================

class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.norm1 = group_norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = group_norm(cout)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = group_norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = group_norm(cout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(y)))


class CSIEncoder(nn.Module):
    def __init__(self, base=32, c_lat=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, base, 3, padding=1), group_norm(base), nn.GELU())
        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)
        self.down3 = DownBlock(base * 4, base * 4)
        self.head = nn.Conv2d(base * 4, c_lat, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.head(x)


class CSIDecoder(nn.Module):
    def __init__(self, base=32, c_lat=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c_lat, base * 4, 3, padding=1), group_norm(base * 4), nn.GELU())
        self.up1 = UpBlock(base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2)
        self.up3 = UpBlock(base * 2, base)
        self.head = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.GELU(), nn.Conv2d(base, 2, 1))

    def forward(self, z):
        return self.head(self.up3(self.up2(self.up1(self.stem(z)))))


# =============================================================================
# SAMM
# =============================================================================

class SAMM(nn.Module):
    def __init__(self, c_lat=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_lat, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))


# =============================================================================
# EMA Vector Quantizer
# =============================================================================

class EMAVectorQuantizer(nn.Module):
    def __init__(self, dim=16, num_embeddings=256, decay=0.95, commitment=0.05):
        super().__init__()
        self.dim = int(dim)
        self.num_embeddings = int(num_embeddings)
        self.decay = float(decay)
        self.commitment = float(commitment)

        self.embedding = nn.Embedding(self.num_embeddings, self.dim)
        self.embedding.weight.requires_grad_(False)

        self.register_buffer("ema_count", torch.ones(self.num_embeddings))
        self.register_buffer("ema_weight", torch.zeros(self.num_embeddings, self.dim))
        self.initialized = False

    @torch.no_grad()
    def initialize_from_data(self, z):
        flat = z.reshape(-1, self.dim)
        if flat.shape[0] >= self.num_embeddings:
            idx = torch.randperm(flat.shape[0], device=flat.device)[:self.num_embeddings]
        else:
            idx = torch.randint(0, flat.shape[0], (self.num_embeddings,), device=flat.device)

        seeds = flat[idx].float()
        self.embedding.weight.copy_(seeds)
        self.ema_weight.copy_(seeds)
        self.ema_count.fill_(1.0)
        self.initialized = True

    @torch.no_grad()
    def _ema_update(self, flat, indices):
        onehot = F.one_hot(indices, self.num_embeddings).float()
        count = onehot.sum(dim=0)
        weight = onehot.t() @ flat.float()

        self.ema_count.mul_(self.decay).add_(count, alpha=1.0 - self.decay)
        self.ema_weight.mul_(self.decay).add_(weight, alpha=1.0 - self.decay)

        total = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-3) / (total + self.num_embeddings * 1e-3) * total
        updated = self.ema_weight / smoothed.unsqueeze(1).clamp_min(1e-6)
        self.embedding.weight.copy_(updated)

    def forward(self, z):
        if not self.initialized:
            if not self.training:
                raise RuntimeError(
                    "VQ codebook is not marked initialized in eval mode. "
                    "After loading a trained checkpoint, set model.vq.initialized = True."
                )
            self.initialize_from_data(z.detach())

        B, N, C = z.shape
        flat = z.reshape(-1, C)
        emb = self.embedding.weight

        dist = (
            flat.float().square().sum(dim=1, keepdim=True)
            + emb.float().square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat.float() @ emb.float().t()
        )

        indices = dist.argmin(dim=1)
        z_q = self.embedding(indices).view(B, N, C).to(z.dtype)

        if self.training:
            self._ema_update(flat.detach(), indices.detach())

        z_st = z + (z_q - z).detach()
        commitment = F.mse_loss(z, z_q.detach())

        with torch.no_grad():
            hist = torch.bincount(indices, minlength=self.num_embeddings).float()
            p = hist / hist.sum().clamp_min(1.0)
            p = p[p > 0]
            perplexity = torch.exp(-(p * torch.log(p)).sum())

        return z_st, indices.view(B, N), commitment, perplexity


# =============================================================================
# ADM: known-token constrained latent completion
# =============================================================================

class ADMBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.ff = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x, known_mask):
        q = self.norm1(x)
        y, _ = self.attn(q, q, q, key_padding_mask=~known_mask, need_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x))


class ADM(nn.Module):
    def __init__(self, num_tokens=16, c_lat=16, d_model=128, layers=4, heads=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_proj = nn.Linear(c_lat, d_model)
        self.out_proj = nn.Linear(d_model, c_lat)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.known_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([ADMBlock(d_model, heads) for _ in range(layers)])

    def forward(self, selected, indices, known_mask):
        B, keep_k, C = selected.shape

        scattered = selected.new_zeros(B, self.num_tokens, C)
        scattered.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, C), selected)

        known_features = self.in_proj(scattered)
        x = torch.where(
            known_mask.unsqueeze(-1),
            known_features + self.known_embed,
            self.mask_token.expand(B, self.num_tokens, -1) + self.missing_embed,
        )
        x = x + self.pos

        for block in self.blocks:
            x = block(x, known_mask)

        recon = self.out_proj(x)
        recon = torch.where(known_mask.unsqueeze(-1), scattered, recon)
        return recon


# =============================================================================
# Complete no-prior TUB-style SSCC model
# =============================================================================

class TUBVQSSCC(nn.Module):
    def __init__(self, base=32, c_lat=16, num_embeddings=256, keep_k=6):
        super().__init__()
        self.c_lat = c_lat
        self.num_tokens = 16
        self.keep_k = keep_k

        self.encoder = CSIEncoder(base, c_lat)
        self.decoder = CSIDecoder(base, c_lat)
        self.samm = SAMM(c_lat)
        self.token_norm = nn.LayerNorm(c_lat)
        self.vq = EMAVectorQuantizer(c_lat, num_embeddings, decay=0.95, commitment=0.05)
        self.adm = ADM(num_tokens=16, c_lat=c_lat, d_model=128, layers=4, heads=4)

    def encode_tokens(self, x):
        z = self.encoder(x)
        B, C, H, W = z.shape
        tokens = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.token_norm(tokens)
        return tokens, (H, W)

    def decode_tokens(self, tokens, hw):
        B, N, C = tokens.shape
        H, W = hw
        z = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z)

    def autoencoder(self, x):
        tokens, hw = self.encode_tokens(x)
        return self.decode_tokens(tokens, hw)

    def select_random(self, B, device):
        noise = torch.rand(B, self.num_tokens, device=device)
        idx = torch.topk(noise, self.keep_k, dim=1).indices
        return idx.sort(dim=1).values

    def select_samm(self, tokens, hw):
        B, N, C = tokens.shape
        H, W = hw
        z = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        score = self.samm(z).flatten(1)
        idx = torch.topk(score, self.keep_k, dim=1, largest=True).indices
        idx = idx.sort(dim=1).values
        return idx, score

    def sparse_forward(self, x, random_mask=False):
        tokens, hw = self.encode_tokens(x)
        B, N, C = tokens.shape

        if random_mask:
            idx = self.select_random(B, x.device)
            score = None
            selected = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        else:
            idx, score = self.select_samm(tokens, hw)
            selected = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))

            score_sel = score.gather(1, idx).unsqueeze(-1)
            selected = selected * (1.0 + score_sel)

        selected_q, code_idx, commitment, perplexity = self.vq(selected)

        known_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        known_mask.scatter_(1, idx, True)

        recon_tokens = self.adm(selected_q, idx, known_mask)
        x_hat = self.decode_tokens(recon_tokens, hw)

        latent_mse = ((recon_tokens - tokens.detach()).square()).mean()

        return x_hat, {
            "tokens": tokens,
            "recon_tokens": recon_tokens,
            "selected_positions": idx,
            "code_indices": code_idx,
            "known_mask": known_mask,
            "commitment": commitment,
            "perplexity": perplexity,
            "latent_mse": latent_mse,
        }


# =============================================================================
# Channel / downstream evaluation
# =============================================================================

def channel_to_images(H_dl, nc):
    H_ad = angular_delay_transform(H_dl, nc)
    B, K = H_ad.shape[:2]
    return H_ad.reshape(B * K, 2, nc, H_dl.shape[2])


def images_to_channel(x_hat, B, K, nsc):
    H_ad_hat = x_hat.view(B, K, *x_hat.shape[1:])
    return inverse_angular_delay_transform(H_ad_hat, nsc)


@torch.no_grad()
def validate(model, generator, args, device):
    model.eval()

    rates = []
    nmses = []
    ae_rates = []
    perplexities = []
    dl_noise = 10.0 ** (-args.dl_snr / 10.0)

    for batch_idx in range(args.val_batches):
        base_eval.set_seed(args.seed + 90_000_000 + batch_idx)
        H_dl, _ = generator.generate_batch_data(args.val_batch_size)
        H_dl = H_dl.to(device)

        B, K = H_dl.shape[:2]
        x = channel_to_images(H_dl, args.nc)

        x_hat, aux = model.sparse_forward(x, random_mask=False)
        H_hat = images_to_channel(x_hat, B, K, args.nsc)

        W = closed_form_rzf(H_hat, args.rzf_reg, args.total_power)
        rates.append(base_eval.per_sample_sum_rate(W, H_dl, dl_noise).cpu())
        nmses.append(base_eval.per_sample_nmse_db(H_hat, H_dl).cpu())
        perplexities.append(aux["perplexity"].detach().cpu())

        x_ae = model.autoencoder(x)
        H_ae = images_to_channel(x_ae, B, K, args.nsc)
        W_ae = closed_form_rzf(H_ae, args.rzf_reg, args.total_power)
        ae_rates.append(base_eval.per_sample_sum_rate(W_ae, H_dl, dl_noise).cpu())

    rate, rate_se = mean_se(rates)
    nmse = torch.cat(nmses).mean().item()
    ae_rate, _ = mean_se(ae_rates)
    perplexity = torch.stack(perplexities).mean().item()
    return rate, rate_se, nmse, ae_rate, perplexity


def set_trainable(module, value):
    for p in module.parameters():
        p.requires_grad = value


def save_checkpoint(path, model, step, stage, val_rate, val_nmse, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "step": step,
        "stage": stage,
        "val_rate": val_rate,
        "val_nmse_db": val_nmse,
        "config": vars(args),
    }, path)


# =============================================================================
# Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nsc", type=int, default=128)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--budget", type=int, default=128)

    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--c-lat", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=256)
    parser.add_argument("--keep-k", type=int, default=6)

    parser.add_argument("--stage1-steps", type=int, default=1500)
    parser.add_argument("--stage2-steps", type=int, default=1500)
    parser.add_argument("--stage3-steps", type=int, default=3000)

    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=250)

    parser.add_argument("--lr-stage1", type=float, default=1e-4)
    parser.add_argument("--lr-adm", type=float, default=2e-4)
    parser.add_argument("--lr-stage3", type=float, default=5e-5)
    parser.add_argument("--lr-samm", type=float, default=2e-4)

    parser.add_argument("--lambda-latent", type=float, default=0.1)
    parser.add_argument("--lambda-commit", type=float, default=0.05)

    parser.add_argument("--dl-snr", type=float, default=25.0)
    parser.add_argument("--rzf-reg", type=float, default=1e-3)
    parser.add_argument("--total-power", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs_digital_baselines/checkpoints/tub_vqsscc_K8_Nsc128_D128_b61_seed42")
    args = parser.parse_args()
    base_eval.set_seed(args.seed)

    if args.nc != 32 or args.nt != 32:
        raise ValueError("This first TWC port assumes Nc=Nt=32.")
    if args.budget % args.k != 0:
        raise ValueError("Dtot must be divisible by K.")

    pos_bits, index_bits, source_bits = combinatorial_source_bits(16, args.keep_k, args.num_embeddings)
    uses_per_ue = args.budget // args.k

    print("=" * 104)
    print("TUB-style no-prior VQ-SSCC baseline")
    print(f"K={args.k}, Nt={args.nt}, Nsc={args.nsc}, Nc={args.nc}, Dtot={args.budget}")
    print(f"Input image: [2,{args.nc},{args.nt}] -> latent grid 4x4 -> 16 tokens")
    print(f"VQ: J={args.num_embeddings}, bits/index={int(math.log2(args.num_embeddings))}, keep_k={args.keep_k}")
    print(f"Combinatorial position bits={pos_bits}, VQ-index bits={index_bits}, source bits/UE={source_bits}")
    print(f"Available feedback uses/UE={uses_per_ue}")
    print(f"With Rc=2/3 + 64-QAM: <= {uses_per_ue} symbols/UE after coded-bit padding")
    print("=" * 104)

    if source_bits > 64:
        raise ValueError(f"Source budget exceeds 64 bits/UE: {source_bits}")

    device = setup_gpu()

    scenario = base_eval.Scenario(
        k_users=args.k,
        subcarriers=args.nsc,
        feedback_budget=args.budget,
        antennas=args.nt,
        carrier_freq=3.5e9,
        speed=1.0,
        total_power=args.total_power,
        num_subbands=4,
    )

    generator = base_eval.build_channel_generator(scenario, device)
    model = TUBVQSSCC(args.base, args.c_lat, args.num_embeddings, args.keep_k).to(device)

    ue_params = (
        sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.samm.parameters())
        + model.vq.embedding.weight.numel()
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"UE-side params (Encoder+SAMM+VQ codebook): {ue_params/1e6:.3f} M")
    print(f"Total trainable/model params: {total_params/1e6:.3f} M")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_nmse = float("inf")
    best_rate = -float("inf")
    global_step = 0

    # -------------------------------------------------------------------------
    # Stage 1: autoencoder pretraining
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Encoder + Decoder autoencoder pretraining")

    set_trainable(model, False)
    set_trainable(model.encoder, True)
    set_trainable(model.decoder, True)

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=args.lr_stage1,
        weight_decay=1e-5,
    )

    progress = tqdm(range(1, args.stage1_steps + 1), desc="TUB-SSCC stage1", dynamic_ncols=True)

    for step in progress:
        global_step += 1
        model.train()

        base_eval.set_seed(args.seed + global_step)
        H_dl, _ = generator.generate_batch_data(args.batch_size)
        x = channel_to_images(H_dl.to(device), args.nc)

        x_hat = model.autoencoder(x)
        loss = nmse_ratio(x_hat, x).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], args.grad_clip)
        optimizer.step()

        if step % 25 == 0:
            progress.set_postfix(nmse=f"{10*math.log10(max(loss.item(),1e-12)):.2f}dB")

    # Initialize the VQ codebook from real Stage-1 latents.
    model.eval()
    with torch.no_grad():
        H_dl, _ = generator.generate_batch_data(args.batch_size)
        x = channel_to_images(H_dl.to(device), args.nc)
        tokens, _ = model.encode_tokens(x)
        model.vq.initialize_from_data(tokens)

    # -------------------------------------------------------------------------
    # Stage 2: random masking + VQ + ADM
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Frozen Encoder/Decoder + random masking + VQ/ADM")

    set_trainable(model, False)
    set_trainable(model.adm, True)

    optimizer = torch.optim.AdamW(model.adm.parameters(), lr=args.lr_adm, weight_decay=1e-5)
    progress = tqdm(range(1, args.stage2_steps + 1), desc="TUB-SSCC stage2", dynamic_ncols=True)

    for step in progress:
        global_step += 1
        model.train()
        model.encoder.eval()
        model.decoder.eval()

        base_eval.set_seed(args.seed + global_step)
        H_dl, _ = generator.generate_batch_data(args.batch_size)
        x = channel_to_images(H_dl.to(device), args.nc)

        with torch.no_grad():
            target_tokens, _ = model.encode_tokens(x)

        x_hat, aux = model.sparse_forward(x, random_mask=True)

        recon_loss = nmse_ratio(x_hat, x).mean()
        latent_loss = aux["latent_mse"]
        loss = recon_loss + args.lambda_latent * latent_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adm.parameters(), args.grad_clip)
        optimizer.step()

        if step % 25 == 0:
            progress.set_postfix(
                nmse=f"{10*math.log10(max(recon_loss.item(),1e-12)):.2f}dB",
                lat=f"{latent_loss.item():.4f}",
                ppl=f"{aux['perplexity'].item():.1f}",
            )

    # -------------------------------------------------------------------------
    # Stage 3: SAMM top-k end-to-end reconstruction fine-tuning
    # -------------------------------------------------------------------------
    print("\n[Stage 3] SAMM top-k + end-to-end fine-tuning")

    set_trainable(model, True)
    model.vq.embedding.weight.requires_grad_(False)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_stage3},
        {"params": model.decoder.parameters(), "lr": args.lr_stage3},
        {"params": model.adm.parameters(), "lr": args.lr_stage3},
        {"params": model.samm.parameters(), "lr": args.lr_samm},
        {"params": model.token_norm.parameters(), "lr": args.lr_stage3},
    ], weight_decay=1e-5)

    progress = tqdm(range(1, args.stage3_steps + 1), desc="TUB-SSCC stage3", dynamic_ncols=True)

    for step in progress:
        global_step += 1
        model.train()

        base_eval.set_seed(args.seed + global_step)
        H_dl, _ = generator.generate_batch_data(args.batch_size)
        x = channel_to_images(H_dl.to(device), args.nc)

        x_hat, aux = model.sparse_forward(x, random_mask=False)

        recon_loss = nmse_ratio(x_hat, x).mean()
        latent_loss = aux["latent_mse"]
        commit_loss = aux["commitment"]

        loss = (
            recon_loss
            + args.lambda_latent * latent_loss
            + args.lambda_commit * commit_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 25 == 0:
            progress.set_postfix(
                nmse=f"{10*math.log10(max(recon_loss.item(),1e-12)):.2f}dB",
                ppl=f"{aux['perplexity'].item():.1f}",
            )

        if step % args.val_interval == 0 or step == args.stage3_steps:
            rate, rate_se, nmse_db, ae_rate, ppl = validate(model, generator, args, device)

            print(
                f"\n[VAL step {step}] "
                f"SSCC rate={rate:.3f} ± {rate_se:.3f}, "
                f"NMSE={nmse_db:.3f} dB, "
                f"full-AE rate={ae_rate:.3f}, "
                f"VQ perplexity={ppl:.1f}/{args.num_embeddings}"
            )

            save_checkpoint(
                output_dir / "latest.pth",
                model, global_step, 3, rate, nmse_db, args
            )

            if nmse_db < best_nmse:
                best_nmse = nmse_db
                save_checkpoint(
                    output_dir / "best_nmse.pth",
                    model, global_step, 3, rate, nmse_db, args
                )
                print(f"[BEST-NMSE] {best_nmse:.3f} dB, rate={rate:.3f}")

            if rate > best_rate:
                best_rate = rate
                save_checkpoint(
                    output_dir / "best_rate.pth",
                    model, global_step, 3, rate, nmse_db, args
                )
                print(f"[BEST-RATE] {best_rate:.3f} bps/Hz, NMSE={nmse_db:.3f} dB")

    print("\nTraining finished.")
    print(f"Best NMSE: {best_nmse:.3f} dB")
    print(f"Best validation rate: {best_rate:.3f} bps/Hz")
    print(f"NMSE checkpoint: {output_dir / 'best_nmse.pth'}")
    print(f"Rate checkpoint: {output_dir / 'best_rate.pth'}")


if __name__ == "__main__":
    main()