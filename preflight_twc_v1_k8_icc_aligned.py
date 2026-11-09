"""Explicit CUDA admission for the new ICC-aligned components; no updates.

This is a synthetic four-MU-frame check, not a CE calibration, benchmark
training run or performance evaluation. Training throughput/ETA is measured
by the included first100 actual optimizer updates, not inferred here.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import time
import traceback

from configs import twc_v1_k8_icc_aligned_config as cfg


def plan():
    return dict(namespace=cfg.NAMESPACE,device="CUDA required;no fallback",synthetic_MU_frames=4,
        full_model_local_execution_allowed=False,optimizer_updates=0,
        checks=["common initialization/independent storage","rank8 actual-pilot Gram/ridge",
            "pilot-only CE including nonfinite data perturbation","one MAC noise/paid energy",
            "strict fresh state roundtrip","both layouts finite DL NMSE backward",
            "two Joint/self/cross, spatial cross and four Swin execution",
            "finite reachable encoder/decoder gradients","frozen CE/EMA","normalized RZF power"],
        precision="FP32/complex64 fixed protocol",performance_claim=False)


def validate_common_initialization(a,b):
    """Works with tiny modules in local tests; full instances only on server."""
    import torch
    aa,bb = dict(a.named_parameters()),dict(b.named_parameters())
    if not aa or set(aa) != set(bb):
        raise ValueError("Common encoder/decoder parameter names differ")
    for key,value in aa.items():
        other = bb[key]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value,other):
            raise ValueError("Common initial parameter differs: "+key)
        if value.untyped_storage().data_ptr() == other.untyped_storage().data_ptr():
            raise ValueError("Two arms share parameter storage: "+key)
    storage_a = {v.untyped_storage().data_ptr() for v in list(a.parameters())+list(a.buffers()) if v.numel()}
    storage_b = {v.untyped_storage().data_ptr() for v in list(b.parameters())+list(b.buffers()) if v.numel()}
    if storage_a & storage_b:
        raise ValueError("Two arms share model/buffer storage")
    return dict(parameter_tensors=len(aa),parameters=sum(v.numel() for v in aa.values()),
        common_values_equal=True,all_storage_independent=True)


def _gradient_record(module):
    import torch
    values = [(key,value.grad) for key,value in module.named_parameters() if value.requires_grad]
    present = [(key,grad) for key,grad in values if grad is not None]
    if not present or any(not bool(torch.isfinite(grad).all()) for _,grad in present):
        raise ValueError("Missing or nonfinite module gradient")
    square = sum(float(grad.detach().double().square().sum()) for _,grad in present)
    if square <= 0:
        raise ValueError("Entire key path has zero gradient")
    return dict(tensors_with_gradient=len(present),trainable_tensors=len(values),L2_norm=square**.5,
        missing=[key for key,grad in values if grad is None],all_present_gradients_finite=True)


def _hook_counts(model):
    counts = {}
    handles = []
    modules = {"encoder":model.encoder,"observation_memory":model.decoder.memory_y,
        "reconstruction_latent":model.decoder.residual_head,"FC":model.decoder.fc,
        "spatial_cross":model.decoder.spatial_attention}
    for i,layer in enumerate(model.decoder.joint_layers):
        modules["joint_self_"+str(i)] = layer.self_attention
        modules["joint_cross_"+str(i)] = layer.cross_attention
    modules.update({"swin_"+str(i):block for i,block in enumerate(model.decoder.blocks)})
    for name,module in modules.items():
        counts[name] = 0
        def hook(_module,_args,_out,key=name):
            counts[key] += 1
        handles.append(module.register_forward_hook(hook))
    return counts,handles


def run_preflight(output):
    """Controller entry: new owned directory, evidence preserved on failure."""
    import train_twc_v1_k8_icc_aligned as rt
    output = rt.inside(output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(plan(),status="running",created=rt.now(),checks={},core=rt.core_identity(),
        execution_code=rt.source_identity(),synthetic_calibration=True,
        synthetic_calibration_description="fixed nonnegative0.02 CE-error feature for admission only;not a saved calibration")
    start = time.perf_counter()
    rng = None
    try:
        device = rt.configure_cuda()
        torch,_ = rt.tensor_runtime()
        rng = rt.capture_rng()
        phy = rt.physics()
        from models.twc_v1_k8_icc_aligned import K8ICCAlignedJointDecoder
        public_args = set(inspect.signature(K8ICCAlignedJointDecoder.forward).parameters)
        if public_args & {"H_dl","H_ul","tx","true_codeword","key_energy","shared_energy"}:
            raise ValueError("Decoder admits simulator truth/current-frame side information")
        report["environment"] = rt.environment()
        a = rt.new_model("fdma","cpu")
        b = rt.new_model("hybrid_g4","cpu")
        report["checks"]["initialization"] = validate_common_initialization(a,b)
        report["parameters"] = {"fdma":a.metadata(),"hybrid_g4":b.metadata()}
        ce = rt.new_model("ce",device).eval().requires_grad_(False)
        ce_digest = rt.state_digest(ce)
        generator = torch.Generator(device=device).manual_seed(cfg.SEED+731)
        def complex_normal(shape):
            return torch.view_as_complex(torch.randn(*shape,2,device=device,generator=generator).contiguous())/2**.5
        H_dl = complex_normal((4,8,32,256))
        H_ul = complex_normal((4,8,32,256))
        for value in (H_dl,H_ul):
            value.div_(value.abs().square().mean((-2,-1),keepdim=True).sqrt())
        noise = complex_normal((4,32,256))
        gamma = torch.tensor([0.,5.,10.,25.],device=device)
        polarity = (2*torch.randint(2,(4,8),generator=generator,device=device)-1).float()
        pilots = phy.known_pilots(4,device=device,polarity=polarity)
        fourier = phy.fourier_diagnostics(pilots)
        if not bool((fourier["design_rank"]==64).all()):
            raise ValueError("Actual eight-pilot per-UE design is not rank64 overall")
        diagonal_error = float((fourier["gram_diagonal"]-16/256).abs().max())
        if diagonal_error > 1e-6 or float(fourier["gram_max_off_diagonal"].max()) > 1e-6:
            raise ValueError("Fourier Gram normalization/ridge scale changed")
        report["checks"]["Fourier_LS"] = dict(rank=fourier["design_rank"].cpu().tolist(),
            design_shape=fourier["design_shape"],gram_diagonal_max_error=diagonal_error,
            off_diagonal_max=float(fourier["gram_max_off_diagonal"].max()),ridge=fourier["ridge"],
            ridge_to_diagonal=fourier["ridge_to_diagonal"])
        Y_p = phy.transmit(H_ul,pilots,noise,gamma)
        initial = phy.fourier_ls(Y_p,pilots)
        with torch.no_grad():
            estimate = ce(Y_p,pilots,initial,gamma_db=gamma)
            poisoned = Y_p.clone()
            poisoned[:,:,~phy.pilot_mask(device).any(0)] = complex(float("nan"),float("inf"))
            poisoned_initial = phy.fourier_ls(poisoned,pilots)
            poisoned_estimate = ce(poisoned,pilots,poisoned_initial,gamma_db=gamma)
        if not torch.equal(initial,poisoned_initial) or not torch.equal(estimate,poisoned_estimate):
            raise ValueError("Data RE affected Fourier-LS or new pilot-only CE")
        report["checks"]["pilot_only_CE"] = dict(data_nan_inf_ignored=True,full_output=list(estimate.shape),
            projected_to_rank8=False,CE_parameters=ce.metadata()["trainable_parameters"])
        # Two CPU instances established independence; move/train only one GPU
        # instance at a time. No optimizer is created and no weight is updated.
        model_states = {"fdma":a,"hybrid_g4":b}
        del a,b
        for arm in cfg.ARMS:
            model = model_states.pop(arm).to(device).train()
            try:
                initial_cpu = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                rt.strict_load(model,initial_cpu)
                layout = phy.make_layout(arm)
                stats = phy.ReceiverStatistics(layout,torch.full((6,8,256),.02)).to(device).train()
                stat_before = rt.state_digest(stats)
                counts,handles = _hook_counts(model)
                torch.cuda.reset_peak_memory_stats()
                raw = model.encode(H_dl)
                wave = phy.build_waveform(raw,layout,polarity)
                Y = phy.transmit(H_ul,wave.x_grid,noise,gamma)
                expected_noise = noise*phy.noise_variance(gamma,4,device).sqrt()[:,None,None]
                actual_noise = Y-(H_ul*wave.x_grid[:,:,None]).sum(1)
                noise_error = float((expected_noise-actual_noise).abs().max())
                if noise_error > 1e-5:
                    raise ValueError("Single common-MAC noise identity failed")
                ledger = phy.energy_ledger(wave,layout)
                energy_error = float((ledger["total"]-64.).abs().max())
                if energy_error > 2e-4:
                    raise ValueError("Paid energy64 per UE/frame failed")
                with torch.no_grad():
                    init = phy.fourier_ls(Y,wave.pilot)
                    corrected_UL = ce(Y,wave.pilot,init,gamma_db=gamma)
                recovered = phy.recover(Y,corrected_UL,layout,stats.nominal_power(),stats.ce_variance(gamma),gamma)
                result = model.decode(Y,corrected_UL,recovered["soft"],recovered["reliability"],wave.pilot,
                    gamma_db=gamma,layout=layout)
                downlink = rt.load_local("utils/twc_v1_k8_downlink.py","_icc_aligned_preflight_downlink")
                loss = downlink.downlink_nmse(result["H_hat"],H_dl).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite physical DL NMSE")
                loss.backward()
                gradients = {"encoder":_gradient_record(model.encoder),"decoder":_gradient_record(model.decoder),
                    "latent_head":_gradient_record(model.decoder.residual_head),"spatial_cross":_gradient_record(model.decoder.spatial_attention)}
                gradients.update({"joint_"+str(i):_gradient_record(layer) for i,layer in enumerate(model.decoder.joint_layers)})
                gradients.update({"swin_"+str(i):_gradient_record(block) for i,block in enumerate(model.decoder.blocks)})
                if not all(count>0 for count in counts.values()):
                    raise ValueError("A required module was not executed: "+repr(counts))
                with torch.no_grad():
                    W,power = downlink.design_rzf(result["H_hat"].detach(),cfg.FIXED_ALPHA)
                power_error = float((power-1).abs().max())
                if power_error > 1e-5:
                    raise ValueError("RZF per-subcarrier unit power failed")
                if rt.state_digest(stats)!=stat_before or rt.state_digest(ce)!=ce_digest:
                    raise ValueError("CE or prior mutated before any successful optimizer update")
                if rt.state_digest(model)!=rt.state_digest(initial_cpu):
                    raise ValueError("Backward-only admission altered model weights/buffers")
                report["checks"][arm] = dict(status="passed",physical_loss=float(loss.detach()),
                    output_shape=list(result["H_hat"].shape),strict_roundtrip=True,execution=counts,gradients=gradients,
                    paid_energy_max_error=energy_error,single_MAC_noise_max_error=noise_error,
                    RZF_power_max_error=power_error,frozen_CE_prior=True,optimizer_updates=0,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    memory_measurement="syntheticB4 forward/backward;no optimizer state;not training peak or ETA")
            finally:
                for handle in handles if "handles" in locals() else []:
                    handle.remove()
                model.zero_grad(set_to_none=True)
                # Graph-bearing locals otherwise retain the previous full GPU
                # instance until the following arm has already been moved.
                raw=wave=Y=recovered=result=loss=W=stats=initial_cpu=corrected_UL=None
                actual_noise=expected_noise=ledger=None
                del model
                torch.cuda.empty_cache()
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed",error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if rng is not None:
            rt.restore_rng(rng)
        report["elapsed_seconds"] = time.perf_counter()-start
        report["completed"] = rt.now()
        rt.atomic_json(output/"report.json",report,exclusive=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run",action="store_true")
    actions.add_argument("--run",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        import json
        print(json.dumps(plan(),indent=2));return 0
    if not args.run:
        parser.print_help();return 0
    if args.output is None:
        parser.error("--run requires an explicit new owned --output directory")
    try:
        run_preflight(args.output)
    except BaseException as exc:
        print("ICC-aligned CUDA preflight failed: "+str(exc),file=sys.stderr);return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
