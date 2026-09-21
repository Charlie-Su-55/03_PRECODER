#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Full VQ-SSCC digital feedback evaluation.

VQ-SSCC source packet per UE:
    13 bits : rank of 6 selected positions among 16
    48 bits : six 8-bit VQ indices
     3 bits : zero padding
    --------------------------------
    64 LDPC information bits

Digital feedback:
    64 info bits
        -> 5G LDPC (64, 96), rate 2/3
        -> 64-QAM
        -> 16 complex symbols / UE
        -> FDMA UL channel + MRC
        -> soft demapping
        -> LDPC decoding

For K=8:
    8 * 16 = 128 complex channel uses = D_tot.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

import baseline_evaluate as base_eval
from models.baseline_common import closed_form_rzf
from train_tub_vqsscc_baseline import TUBVQSSCC, channel_to_images, images_to_channel
from utils import setup_gpu

from sionna.phy import config as sionna_config
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper


def mean_se(values):
    x = torch.cat(values).float()
    mean = x.mean().item()
    se = (x.std(unbiased=True) / math.sqrt(x.numel())).item()
    return mean, se


def ints_to_bits(x, num_bits):
    shifts = torch.arange(num_bits - 1, -1, -1, device=x.device, dtype=torch.long)
    return ((x.long().unsqueeze(-1) >> shifts) & 1).float()


def bits_to_int(bits):
    num_bits = bits.shape[-1]
    weights = 2 ** torch.arange(
        num_bits - 1, -1, -1, device=bits.device, dtype=torch.long
    )
    return (bits.long() * weights).sum(dim=-1)


class VQSourcePacket:
    def __init__(self, num_tokens, keep_k, num_embeddings, device):
        self.num_tokens = int(num_tokens)
        self.keep_k = int(keep_k)
        self.num_embeddings = int(num_embeddings)
        self.device = device

        if self.num_embeddings & (self.num_embeddings - 1):
            raise ValueError("num_embeddings must be a power of two.")

        self.bits_per_index = int(math.log2(self.num_embeddings))
        self.combinations = list(
            itertools.combinations(range(self.num_tokens), self.keep_k)
        )
        self.num_combinations = len(self.combinations)
        self.position_bits = math.ceil(math.log2(self.num_combinations))
        self.index_bits = self.keep_k * self.bits_per_index
        self.source_bits = self.position_bits + self.index_bits
        self.info_bits = 64

        if self.source_bits > self.info_bits:
            raise ValueError(
                f"Source description requires {self.source_bits} bits, "
                f"but LDPC information block has only {self.info_bits}."
            )

        self.rank_map = {
            tuple(c): rank for rank, c in enumerate(self.combinations)
        }
        self.combo_table = torch.tensor(
            self.combinations, dtype=torch.long, device=device
        )

    def pack(self, positions, vq_indices):
        n = positions.shape[0]

        # Combinatorial coding represents an unordered position set.
        # Canonicalize the positions and apply the same permutation to
        # the associated VQ indices so that position-index pairs are preserved.
        order = torch.argsort(positions, dim=1)
        positions = torch.gather(positions, 1, order)
        vq_indices = torch.gather(vq_indices, 1, order)

        positions_cpu = positions.detach().cpu().tolist()
        ranks = [
            self.rank_map[tuple(int(v) for v in row)]
            for row in positions_cpu
        ]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)

        pos_bits = ints_to_bits(ranks, self.position_bits)
        idx_bits = ints_to_bits(
            vq_indices.reshape(-1),
            self.bits_per_index,
        ).view(n, -1)

        source = torch.cat([pos_bits, idx_bits], dim=-1)

        pad_bits = self.info_bits - source.shape[-1]
        if pad_bits:
            source = torch.cat(
                [
                    source,
                    torch.zeros(
                        n,
                        pad_bits,
                        device=self.device,
                        dtype=source.dtype,
                    ),
                ],
                dim=-1,
            )

        return source

    def unpack(self, bits):
        rank = bits_to_int(bits[:, :self.position_bits])

        valid = rank < self.num_combinations
        safe_rank = torch.where(valid, rank, torch.zeros_like(rank))
        positions = self.combo_table[safe_rank]

        start = self.position_bits
        stop = start + self.index_bits

        idx_bits = bits[:, start:stop].reshape(
            bits.shape[0],
            self.keep_k,
            self.bits_per_index,
        )
        vq_indices = bits_to_int(idx_bits)

        return positions, vq_indices, valid


class DigitalFeedbackChannel:
    def __init__(self, device):
        device_str = str(device)

        sionna_config.device = device_str

        self.encoder = LDPC5GEncoder(
            k=64,
            n=96,
            num_bits_per_symbol=6,
            device=device_str,
        )

        self.decoder = LDPC5GDecoder(
            self.encoder,
            cn_update="minsum",
            num_iter=20,
            hard_out=True,
            return_infobits=True,
            device=device_str,
        )

        self.mapper = Mapper(
            "qam",
            6,
            device=device_str,
        )

        self.demapper = Demapper(
            "app",
            "qam",
            6,
            device=device_str,
        )

    def encode_and_map(self, info_bits):
        coded = self.encoder(info_bits)
        symbols = self.mapper(coded)

        if coded.shape[-1] != 96:
            raise RuntimeError(
                f"Expected 96 coded bits, got {coded.shape[-1]}."
            )
        if symbols.shape[-1] != 16:
            raise RuntimeError(
                f"Expected 16 64-QAM symbols, got {symbols.shape[-1]}."
            )

        return coded, symbols

    def decode(self, equalized, no_eff):
        llr = self.demapper(equalized, no=no_eff)
        return self.decoder(llr)


def fdma_mrc_digital(symbols, H_ul, snr_db):
    """Digital FDMA transmission using the same UL segmentation/MRC geometry
    as the analog FDMA baseline.

    symbols: [B,K,16]
    H_ul:    [B,K,Nt,Nsc]
    """

    B, K, n_sym = symbols.shape

    if K * n_sym > H_ul.shape[-1]:
        raise ValueError(
            f"Digital FDMA requires {K*n_sym} subcarriers, "
            f"but H_ul has only {H_ul.shape[-1]}."
        )

    H_seg = torch.stack(
        [
            H_ul[:, k, :, k*n_sym:(k+1)*n_sym]
            for k in range(K)
        ],
        dim=1,
    )

    clean = H_seg * symbols.unsqueeze(2)

    snr_lin = 10.0 ** (float(snr_db) / 10.0)

    # Match the received-power-relative SNR convention used by the
    # existing analog FDMA feedback implementation.
    no_rx = clean.abs().square().mean() / snr_lin

    noise = torch.sqrt(no_rx / 2.0) * (
        torch.randn_like(clean.real)
        + 1j * torch.randn_like(clean.real)
    )

    received = clean + noise

    denominator = H_seg.abs().square().sum(dim=2).clamp_min(1e-8)

    equalized = (
        H_seg.conj() * received
    ).sum(dim=2) / denominator

    # Effective complex AWGN variance after MRC.
    no_eff = no_rx / denominator

    return equalized, no_eff


def reconstruct_from_digital(model, positions, vq_indices, nc):
    n = positions.shape[0]

    selected_q = model.vq.embedding(vq_indices)

    known_mask = torch.zeros(
        n,
        model.num_tokens,
        dtype=torch.bool,
        device=positions.device,
    )
    known_mask.scatter_(1, positions, True)

    recon_tokens = model.adm(
        selected_q,
        positions,
        known_mask,
    )

    h_lat = nc // 8
    w_lat = model.num_tokens // h_lat

    return model.decode_tokens(
        recon_tokens,
        (h_lat, w_lat),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--snr-list",
        type=float,
        nargs="+",
        default=[0, 5, 10, 15, 20, 25],
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=200)

    parser.add_argument("--dl-snr", type=float, default=25.0)
    parser.add_argument("--rzf-reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260823)

    parser.add_argument(
        "--output-dir",
        default="runs_evaluation/vqsscc/k8_n128_d128/digital_feedback",
    )

    args = parser.parse_args()

    device = setup_gpu()
    sionna_config.device = str(device)
    sionna_config.seed = args.seed

    checkpoint_path = Path(args.checkpoint)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint["config"]

    k = int(cfg["k"])
    nt = int(cfg["nt"])
    nsc = int(cfg["nsc"])
    nc = int(cfg["nc"])
    budget = int(cfg["budget"])

    base = int(cfg["base"])
    c_lat = int(cfg["c_lat"])
    num_embeddings = int(cfg["num_embeddings"])
    keep_k = int(cfg["keep_k"])
    total_power = float(cfg["total_power"])

    model = TUBVQSSCC(
        base=base,
        c_lat=c_lat,
        num_embeddings=num_embeddings,
        keep_k=keep_k,
    ).to(device)

    model.load_state_dict(
        checkpoint["state"],
        strict=True,
    )

    # Critical: preserve the trained VQ codebook after checkpoint loading.
    model.vq.initialized = True
    model.eval()

    packet = VQSourcePacket(
        num_tokens=model.num_tokens,
        keep_k=keep_k,
        num_embeddings=num_embeddings,
        device=device,
    )

    digital = DigitalFeedbackChannel(device)

    symbols_per_ue = 96 // 6

    if symbols_per_ue != 16:
        raise RuntimeError("Expected exactly 16 feedback symbols per UE.")

    if k * symbols_per_ue != budget:
        raise ValueError(
            f"Digital feedback uses {k*symbols_per_ue} complex symbols, "
            f"but D_tot={budget}."
        )

    scenario = base_eval.Scenario(
        k_users=k,
        subcarriers=nsc,
        feedback_budget=budget,
        antennas=nt,
        carrier_freq=3.5e9,
        speed=1.0,
        total_power=total_power,
        num_subbands=4,
    )

    generator = base_eval.build_channel_generator(
        scenario,
        device,
    )

    print("=" * 104)
    print("VQ-SSCC full digital feedback evaluation")
    print(f"K={k}, Nt={nt}, Nsc={nsc}, Dtot={budget}")
    print(
        f"Source packet: {packet.position_bits} position bits + "
        f"{packet.index_bits} VQ-index bits = "
        f"{packet.source_bits} source bits/UE"
    )
    print(
        f"LDPC input: {packet.info_bits} bits "
        f"({packet.info_bits-packet.source_bits} zero padding bits)"
    )
    print(
        f"5G LDPC: 64 -> 96 bits, "
        f"64-QAM -> {symbols_per_ue} complex symbols/UE"
    )
    print(
        f"Aggregate feedback uses: "
        f"{k} x {symbols_per_ue} = "
        f"{k*symbols_per_ue} = Dtot"
    )
    print(
        f"DL SNR={args.dl_snr:.1f} dB, "
        f"RZF lambda={args.rzf_reg:g}"
    )
    print("=" * 104)

    dl_noise = 10.0 ** (-args.dl_snr / 10.0)

    ideal_rates = []
    ideal_nmses = []

    rate_lists = {
        snr: [] for snr in args.snr_list
    }
    nmse_lists = {
        snr: [] for snr in args.snr_list
    }

    bit_errors = {
        snr: 0 for snr in args.snr_list
    }
    bit_totals = {
        snr: 0 for snr in args.snr_list
    }
    ue_block_errors = {
        snr: 0 for snr in args.snr_list
    }
    ue_block_totals = {
        snr: 0 for snr in args.snr_list
    }
    system_block_errors = {
        snr: 0 for snr in args.snr_list
    }
    system_block_totals = {
        snr: 0 for snr in args.snr_list
    }
    invalid_positions = {
        snr: 0 for snr in args.snr_list
    }
    invalid_totals = {
        snr: 0 for snr in args.snr_list
    }

    clean_roundtrip_checked = False

    progress = tqdm(
        range(args.num_batches),
        desc="VQ-SSCC digital",
        dynamic_ncols=True,
    )

    for batch_idx in progress:
        base_eval.set_seed(args.seed + batch_idx)

        H_dl, H_ul = generator.generate_batch_data(
            args.batch_size
        )

        H_dl = H_dl.to(device)
        H_ul = H_ul.to(device)

        B, K = H_dl.shape[:2]

        x = channel_to_images(
            H_dl,
            nc,
        )

        # UE source encoder / importance selector / VQ.
        x_ideal, aux = model.sparse_forward(
            x,
            random_mask=False,
        )

        positions = aux["selected_positions"]
        vq_indices = aux["code_indices"]

        info_tx = packet.pack(
            positions,
            vq_indices,
        )
        # Perfect source-bit delivery reference:
        # exact packet recovery -> codebook lookup -> ADM -> decoder.
        pos_ideal, vq_ideal, valid_ideal = packet.unpack(info_tx)

        if not valid_ideal.all():
            raise RuntimeError(
                "Perfect source-bit packet produced an invalid position rank."
            )

        x_packet_ideal = reconstruct_from_digital(
            model,
            pos_ideal,
            vq_ideal,
            nc,
        )
        # -------------------------------------------------------------
        # Exact clean packet round-trip verification, done once.
        # -------------------------------------------------------------
        if not clean_roundtrip_checked:
            pos_clean, vq_clean, valid_clean = packet.unpack(info_tx)

            # pack() canonicalizes selected positions into ascending order and
            # applies the same permutation to their associated VQ indices.
            order = torch.argsort(positions, dim=1)
            pos_ref = torch.gather(positions, 1, order)
            vq_ref = torch.gather(vq_indices, 1, order)

            positions_exact = torch.equal(pos_clean, pos_ref)
            vq_indices_exact = torch.equal(vq_clean, vq_ref)
            ranks_valid = bool(valid_clean.all().item())

            if not ranks_valid:
                raise RuntimeError(
                    "Clean combinatorial packet round-trip produced an invalid position rank."
                )
            if not positions_exact:
                raise RuntimeError(
                    "Clean packet round-trip changed the selected token positions."
                )
            if not vq_indices_exact:
                raise RuntimeError(
                    "Clean packet round-trip changed the VQ code indices."
                )

            x_clean = reconstruct_from_digital(
                model,
                pos_clean,
                vq_clean,
                nc,
            )

            diff = x_clean - x_ideal
            max_diff = diff.abs().max().item()
            mean_diff = diff.abs().mean().item()
            rel_mse = (
                diff.float().square().sum()
                / x_ideal.float().square().sum().clamp_min(1e-12)
            ).item()

            print(
                "\n[CLEAN ROUNDTRIP]"
                f"\n  position bits exact : {positions_exact}"
                f"\n  VQ-index bits exact : {vq_indices_exact}"
                f"\n  all ranks valid      : {ranks_valid}"
                f"\n  max |packet-ST|      : {max_diff:.3e}"
                f"\n  mean |packet-ST|     : {mean_diff:.3e}"
                f"\n  relative MSE         : {rel_mse:.3e}"
            )

            # The discrete packet identity is the strict correctness criterion.
            # x_ideal uses the straight-through VQ tensor, whereas x_clean uses
            # an explicit codebook lookup, so bitwise equality after the nonlinear
            # ADM/decoder is neither required nor expected.
            if max_diff > 1e-2:
                raise RuntimeError(
                    "Packet bits round-trip exactly, but explicit codebook reconstruction "
                    "differs unexpectedly from the straight-through model path."
                )

            clean_roundtrip_checked = True

        # -------------------------------------------------------------
        # Ideal source-bit delivery reference on the SAME samples.
        # -------------------------------------------------------------
        H_ideal = images_to_channel(
            x_packet_ideal,
            B,
            K,
            nsc,
        )

        W_ideal = closed_form_rzf(
            H_ideal,
            regularization=args.rzf_reg,
            total_power=total_power,
        )

        ideal_rates.append(
            base_eval.per_sample_sum_rate(
                W_ideal,
                H_dl,
                dl_noise,
            ).cpu()
        )

        ideal_nmses.append(
            base_eval.per_sample_nmse_db(
                H_ideal,
                H_dl,
            ).cpu()
        )

        # LDPC encoding and QAM mapping are independent of SNR.
        _, mapped = digital.encode_and_map(
            info_tx
        )

        mapped = mapped.view(
            B,
            K,
            symbols_per_ue,
        )

        # -------------------------------------------------------------
        # Feedback-SNR sweep
        # -------------------------------------------------------------
        for snr_idx, snr_db in enumerate(args.snr_list):
            base_eval.set_seed(
                args.seed
                + 10_000_000
                + 100_000 * snr_idx
                + batch_idx
            )

            equalized, no_eff = fdma_mrc_digital(
                mapped,
                H_ul,
                snr_db,
            )

            info_hat = digital.decode(
                equalized.reshape(
                    B * K,
                    symbols_per_ue,
                ),
                no_eff.reshape(
                    B * K,
                    symbols_per_ue,
                ),
            )

            info_hat = (
                info_hat > 0.5
            ).float()

            # Metrics only on the 61 meaningful source bits.
            tx_source = info_tx[:, :packet.source_bits]
            rx_source = info_hat[:, :packet.source_bits]

            bit_error_mask = (
                tx_source != rx_source
            )

            bit_errors[snr_db] += int(
                bit_error_mask.sum().item()
            )
            bit_totals[snr_db] += int(
                bit_error_mask.numel()
            )

            ue_error = bit_error_mask.any(dim=-1)

            ue_block_errors[snr_db] += int(
                ue_error.sum().item()
            )
            ue_block_totals[snr_db] += int(
                ue_error.numel()
            )

            system_error = ue_error.view(
                B,
                K,
            ).any(dim=1)

            system_block_errors[snr_db] += int(
                system_error.sum().item()
            )
            system_block_totals[snr_db] += B

            pos_hat, vq_hat, valid = packet.unpack(
                info_hat
            )

            invalid_positions[snr_db] += int(
                (~valid).sum().item()
            )
            invalid_totals[snr_db] += int(
                valid.numel()
            )

            x_hat = reconstruct_from_digital(
                model,
                pos_hat,
                vq_hat,
                nc,
            )

            H_hat = images_to_channel(
                x_hat,
                B,
                K,
                nsc,
            )

            W = closed_form_rzf(
                H_hat,
                regularization=args.rzf_reg,
                total_power=total_power,
            )

            rate_lists[snr_db].append(
                base_eval.per_sample_sum_rate(
                    W,
                    H_dl,
                    dl_noise,
                ).cpu()
            )

            nmse_lists[snr_db].append(
                base_eval.per_sample_nmse_db(
                    H_hat,
                    H_dl,
                ).cpu()
            )

    ideal_rate, ideal_se = mean_se(
        ideal_rates
    )
    ideal_nmse = torch.cat(
        ideal_nmses
    ).mean().item()

    print("\n" + "=" * 118)
    print(
        f"Ideal VQ-SSCC source delivery: "
        f"rate={ideal_rate:.3f} ± {ideal_se:.3f} bps/Hz, "
        f"NMSE={ideal_nmse:.3f} dB"
    )
    print("=" * 118)

    header = (
        f"{'FB SNR':>8} "
        f"{'Rate':>10} "
        f"{'SE':>8} "
        f"{'NMSE(dB)':>10} "
        f"{'BER':>12} "
        f"{'UE-BLER':>12} "
        f"{'SYS-BLER':>12} "
        f"{'InvalidPos':>12}"
    )

    print(header)
    print("-" * 118)

    rows = []

    for snr_db in args.snr_list:
        rate_mean, rate_se = mean_se(
            rate_lists[snr_db]
        )
        nmse = torch.cat(
            nmse_lists[snr_db]
        ).mean().item()

        ber = (
            bit_errors[snr_db]
            / max(bit_totals[snr_db], 1)
        )
        ue_bler = (
            ue_block_errors[snr_db]
            / max(ue_block_totals[snr_db], 1)
        )
        sys_bler = (
            system_block_errors[snr_db]
            / max(system_block_totals[snr_db], 1)
        )
        invalid_rate = (
            invalid_positions[snr_db]
            / max(invalid_totals[snr_db], 1)
        )

        print(
            f"{snr_db:>8.1f} "
            f"{rate_mean:>10.3f} "
            f"{rate_se:>8.3f} "
            f"{nmse:>10.3f} "
            f"{ber:>12.3e} "
            f"{ue_bler:>12.3e} "
            f"{sys_bler:>12.3e} "
            f"{invalid_rate:>12.3e}"
        )

        rows.append({
            "feedback_snr_db": snr_db,
            "sum_rate_mean": rate_mean,
            "sum_rate_se": rate_se,
            "nmse_db": nmse,
            "source_ber": ber,
            "ue_bler": ue_bler,
            "system_bler": sys_bler,
            "invalid_position_rate": invalid_rate,
            "ideal_sum_rate_mean": ideal_rate,
            "ideal_sum_rate_se": ideal_se,
            "ideal_nmse_db": ideal_nmse,
            "source_bits_per_ue": packet.source_bits,
            "ldpc_info_bits": 64,
            "ldpc_coded_bits": 96,
            "qam_order": 64,
            "symbols_per_ue": symbols_per_ue,
            "aggregate_channel_uses": k * symbols_per_ue,
            "rzf_reg": args.rzf_reg,
            "num_samples": args.batch_size * args.num_batches,
        })

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "vqsscc_digital_feedback.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 118)
    print(f"Saved: {output_path}")
    print("=" * 118)


if __name__ == "__main__":
    main()