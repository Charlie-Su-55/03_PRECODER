#!/usr/bin/env python3
"""Validation-only K8 feedback-link decomposition; CPU default, no updates/test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from configs import twc_v1_k8_codec_warmup_config as cfg

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs/twc_v1_k8_codec_warmup/paired_link_diagnostics_cpu_20261106"


def load_local(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


system = load_local("utils/twc_v1_k8_system.py", "_twc_v1_k8_link_diag_system")
data_module = load_local("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_link_diag_data")
protocol = system.protocol
data_class = data_module.TWCv1K8ReplayableData


def noise(batch: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((batch, 32, 256, 2), generator=gen, dtype=torch.float32)
    return torch.view_as_complex(value.contiguous())


def nmse_by_sample(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (estimate - target).abs().square().sum(dim=-1).mean(dim=-1)
    denominator = target.abs().square().sum(dim=-1).mean(dim=-1).clamp_min(1e-20)
    return numerator / denominator


def append_symbol_rows(rows, *, sample, batch_index, arm, seed, group_count,
                       policy, gamma, branch, target, recovered, reliability,
                       posterior, energy_target):
    err = nmse_by_sample(recovered, target)
    rel = reliability.mean(dim=(-1, -2))
    post = posterior.mean(dim=(-1, -2))
    actual = target.abs().square().sum(dim=-1).mean(dim=-1)
    for i in range(target.shape[0]):
        rows.append({"batch": batch_index, "sample_in_batch": i,
            "sample_fingerprint": sample["sample_fingerprint"], "arm": arm,
            "seed": seed, "group_count": group_count, "energy_policy": policy,
            "gamma_db": gamma, "pilot_energy_per_ue": 16.0,
            "branch": branch, "symbol_nmse": float(err[i]),
            "symbol_nmse_db": float(10 * torch.log10(err[i].clamp_min(1e-20))),
            "mean_soft_reliability": float(rel[i]),
            "mean_posterior_variance": float(post[i]),
            "target_branch_energy_per_ue": energy_target,
            "actual_tx_branch_energy_per_ue": float(actual[i])})


def rx_power_stats(Hul, tx, layout):
    """Average desired and same-group other-user component powers at the BS."""
    contributions = Hul * tx["X"][:, :, None, :]
    component_power = contributions.abs().square().mean(dim=2)  # [B,K,RE]
    desired_key, desired_shared, interference_shared = [], [], []
    for user, positions in enumerate(layout.key_positions):
        q = torch.tensor(positions, dtype=torch.long)
        desired_key.append(component_power[:, user].index_select(-1, q).mean())
    for members, positions in zip(layout.groups, layout.shared_positions_by_group):
        q = torch.tensor(positions, dtype=torch.long)
        for user in members:
            desired_shared.append(component_power[:, user].index_select(-1, q).mean())
            other = [v for v in members if v != user]
            if other:
                interference_shared.append(component_power[:, other].index_select(-1, q).sum(dim=1).mean())
    return {"key_desired_rx_power": float(torch.stack(desired_key).mean()),
        "shared_desired_rx_power": float(torch.stack(desired_shared).mean()) if desired_shared else 0.0,
        "shared_intragroup_interference_rx_power": float(torch.stack(interference_shared).mean()) if interference_shared else 0.0}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No records for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.batches < 2 or args.batch_size < 1:
        raise ValueError("Use at least two validation batches and a positive batch size")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {out}")
    torch.set_num_threads(1)
    sizes = {key: args.batch_size for key in
        ("ce_train", "codec_train", "ce_calibration", "validation", "test")}
    data = data_class(base_seed=cfg.BASE_DATA_SEED, batch_sizes=sizes, device="cpu",
        cdl_model=cfg.CDL_MODEL, speed=cfg.SPEED_MPS, seed_namespace=cfg.SEED_NAMESPACE)
    chain = system.K8FeedbackChain.from_artifacts(ROOT / cfg.CE_CHECKPOINT,
        ROOT / cfg.CE_CALIBRATION, device="cpu")
    chain.eval()
    runs = [("fdma", 20261106, ROOT / "runs/twc_v1_k8_codec_warmup/fdma/seed_20261106_jointv1"),
        ("fdma", 20261107, ROOT / "runs/twc_v1_k8_codec_warmup/fdma/seed_20261107_jointv1"),
        ("hybrid", 20261106, ROOT / "runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1"),
        ("hybrid", 20261107, ROOT / "runs/twc_v1_k8_codec_warmup/hybrid/seed_20261107_jointv1")]
    ckpt_meta = {}
    codec_states = {}
    for arm, seed, folder in runs:
        path = folder / "checkpoint_latest.pt"
        c = torch.load(path, map_location="cpu", weights_only=False)
        if (c.get("arm"), c.get("seed"), c.get("step")) != (arm, seed, 10000):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        codec_states[(arm, seed)] = c["codec_model"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        ckpt_meta[(arm, seed)] = {"path": str(path), "sha256": digest.hexdigest()}
        del c
    print("Loaded all four verified step-10000 checkpoints; CPU inference only.", flush=True)

    symbol_rows, link_rows = [], []
    samples = []
    for bi in range(args.batches):
        sample = data.sample("validation", bi)
        samples.append({"batch": bi, "sample_fingerprint": sample["sample_fingerprint"]})
        Hdl, Hul = sample["H_dl"], sample["H_ul"]
        B = Hdl.shape[0]
        for gamma_i, gamma in enumerate(cfg.GAMMAS):
            pnoise = noise(B, 50_000_000 + bi * 100_000 + gamma_i * 1000 + 10)
            dnoise = noise(B, 60_000_000 + bi * 100_000 + gamma_i * 1000)
            pilot_y = system.link.transmit_pilots(Hul, 16.0, gamma, pnoise)
            hls = system.link.estimate_ul_ls_interpolated(pilot_y, 16.0)
            hest = chain.ce_model(pilot_y, chain._known_pilots(B, torch.device("cpu")), hls,
                pilot_energy_per_ue=16.0, noise_variance=protocol.feedback_noise_variance(gamma))
            ce_var = chain.calibration.error_variance(gamma, 16.0, device=torch.device("cpu")).expand(B, 8)
            for arm, seed, _ in runs:
                chain.codec_model.load_state_dict(codec_states[(arm, seed)], strict=True)
                if arm == "fdma":
                    layout = protocol.FDMALayout()
                    data_energy = 48.0
                    tx_symbols = chain.codec_model.encode_fdma(Hdl)
                    tx = system.link.transmit_feedback_fdma_data(Hul, layout, data_energy,
                        gamma, tx_symbols, dnoise)
                    rec = system.link.recover_feedback_fdma_data(tx["Y"], hest, layout,
                        data_energy, gamma, ce_var)
                    power = component_power_fdma(Hul, tx, layout)
                    for si in range(B):
                        symbol_rows.append({"batch": bi, "sample_in_batch": si,
                            "sample_fingerprint": sample["sample_fingerprint"], "arm": arm,
                            "seed": seed, "group_count": 1, "energy_policy": "fdma",
                            "gamma_db": gamma, "pilot_energy_per_ue": 16.0,
                            "branch": "fdma_data24", "symbol_nmse": float(nmse_by_sample(rec["data_symbols_hat"], tx["data_symbols_tx"])[si]),
                            "symbol_nmse_db": float(10*torch.log10(nmse_by_sample(rec["data_symbols_hat"], tx["data_symbols_tx"])[si].clamp_min(1e-20))),
                            "mean_soft_reliability": float(rec["reliability"][si].mean()),
                            "mean_posterior_variance": float(rec["posterior_variance"][si].mean()),
                            "target_branch_energy_per_ue": data_energy,
                            "actual_tx_branch_energy_per_ue": float(tx["actual_data_energy"][si].mean())})
                    link_rows.append({"arm": arm, "seed": seed, "group_count": 1,
                        "energy_policy": "fdma", "gamma_db": gamma, "N0": tx["N0"],
                        "data_energy_per_ue": data_energy, "actual_energy_max_abs_error": float((tx["actual_data_energy"]-data_energy).abs().max()),
                        **power})
                else:
                    for groups in (1, 2, 4):
                        layout = protocol.default_hybrid_layout(groups)
                        key, shared = chain.codec_model.encode(Hdl, group_count=groups)
                        for ratio_name, ratio in (("hybrid_1_3", (1.0, 3.0)), ("hybrid_1_1", (1.0, 1.0))):
                            ekey = 48.0 * ratio[0] / sum(ratio)
                            eshared = 48.0 * ratio[1] / sum(ratio)
                            split = protocol.EnergySplit(16.0, ekey, eshared)
                            tx = system.link.transmit_feedback_data(Hul, layout, split, gamma,
                                key, shared, dnoise)
                            rec = system.link.recover_feedback_data(tx["Y"], hest, layout,
                                split, gamma, ce_var)
                            append_symbol_rows(symbol_rows, sample=sample, batch_index=bi,
                                arm=arm, seed=seed, group_count=groups, policy=ratio_name,
                                gamma=gamma, branch="key_d4", target=tx["key_symbols_tx"],
                                recovered=rec["key_symbols_hat"], reliability=rec["key_reliability"],
                                posterior=rec["key_posterior_variance"], energy_target=ekey)
                            append_symbol_rows(symbol_rows, sample=sample, batch_index=bi,
                                arm=arm, seed=seed, group_count=groups, policy=ratio_name,
                                gamma=gamma, branch=f"shared_d{layout.shared_dimension}", target=tx["shared_symbols_tx"],
                                recovered=rec["shared_symbols_hat"], reliability=rec["shared_reliability"],
                                posterior=rec["shared_posterior_variance"], energy_target=eshared)
                            power = rx_power_stats(Hul, tx, layout)
                            link_rows.append({"arm": arm, "seed": seed, "group_count": groups,
                                "energy_policy": ratio_name, "gamma_db": gamma, "N0": tx["N0"],
                                "pilot_energy_per_ue": 16.0, "key_energy_per_ue": ekey,
                                "shared_energy_per_ue": eshared,
                                "actual_total_energy_max_abs_error": float((tx["actual_data_energy"]-48.0).abs().max()),
                                "actual_key_energy_max_abs_error": float((tx["actual_key_energy"]-ekey).abs().max()),
                                "actual_shared_energy_max_abs_error": float((tx["actual_shared_energy"]-eshared).abs().max()),
                                "key_symbol_energy_per_dimension": ekey/4,
                                "shared_symbol_energy_per_dimension": eshared/layout.shared_dimension,
                                **power})
        print(f"Completed link diagnostics batch {bi+1}/{args.batches}.", flush=True)

    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "soft_symbol_diagnostics.csv", symbol_rows)
    write_csv(out / "link_power_energy_diagnostics.csv", link_rows)
    # Summaries retain each SNR because soft recovery is SNR-dependent.
    grouped = {}
    for row in symbol_rows:
        k=(row["arm"],row["seed"],row["group_count"],row["energy_policy"],row["gamma_db"],row["branch"])
        grouped.setdefault(k,[]).append(row)
    summary=[]
    for k, vals in sorted(grouped.items()):
        summary.append({"arm":k[0],"seed":k[1],"group_count":k[2],"energy_policy":k[3],"gamma_db":k[4],"branch":k[5],"n_samples":len(vals),
            "symbol_nmse_mean":sum(v["symbol_nmse"] for v in vals)/len(vals),
            "symbol_nmse_db_of_mean":10*torch.log10(torch.tensor(sum(v["symbol_nmse"] for v in vals)/len(vals))).item(),
            "mean_soft_reliability":sum(v["mean_soft_reliability"] for v in vals)/len(vals),
            "mean_posterior_variance":sum(v["mean_posterior_variance"] for v in vals)/len(vals),
            "actual_tx_branch_energy_mean":sum(v["actual_tx_branch_energy_per_ue"] for v in vals)/len(vals)})
    write_csv(out / "soft_symbol_summary.csv", summary)
    # Combine link power rows across SNR; signal powers are channel/codeword based,
    # while N0 is kept in the row-level CSV.
    link_summary=[]
    lg={}
    for row in link_rows:
        k=(row["arm"],row["seed"],row["group_count"],row["energy_policy"])
        lg.setdefault(k,[]).append(row)
    for k, vals in sorted(lg.items()):
        keys=[name for name in vals[0] if name.endswith("power") or name.endswith("error") or name in ("key_symbol_energy_per_dimension","shared_symbol_energy_per_dimension","pilot_energy_per_ue","key_energy_per_ue","shared_energy_per_ue","data_energy_per_ue")]
        row={"arm":k[0],"seed":k[1],"group_count":k[2],"energy_policy":k[3],"n_snr_rows":len(vals)}
        for name in keys:
            row[name]=sum(float(v[name]) for v in vals)/len(vals)
        link_summary.append(row)
    write_csv(out / "link_power_energy_summary.csv", link_summary)
    manifest={"status":"completed","evaluation":"validation_only_paired_link_decomposition",
        "device":"cpu","test_used":False,"optimizer_updates":0,"samples":args.batches*args.batch_size,
        "batch_size":args.batch_size,"batches":args.batches,"data_protocol":data.protocol,
        "sample_fingerprints":samples,"common_validation_channels":True,"common_pilot_and_data_noise":True,
        "gamma_db":list(cfg.GAMMAS),"pilot_energy_per_ue":16.0,"pilot_re":64,
        "energy_ratios":{"hybrid_1_3":[12.0,36.0],"hybrid_1_1":[24.0,24.0]},
        "checkpoints":{f"{a}_seed{s}":m for (a,s),m in ckpt_meta.items()},
        "interpretation":"Stage-B warmup link diagnostics on validation role only; no independent test evidence."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"status=completed samples={args.batches*args.batch_size} test_used=False updates=0 output={out}",flush=True)


def component_power_fdma(Hul, tx, layout):
    contributions = Hul * tx["X"][:, :, None, :]
    p = contributions.abs().square().mean(dim=2)
    desired=[]
    for user, positions in enumerate(layout.data_positions):
        q=torch.tensor(positions,dtype=torch.long)
        desired.append(p[:,user].index_select(-1,q).mean())
    return {"data_desired_rx_power":float(torch.stack(desired).mean()),"data_intragroup_interference_rx_power":0.0}


if __name__ == "__main__":
    main()
