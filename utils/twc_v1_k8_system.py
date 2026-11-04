"""Differentiable K8 pilot -> feedback -> CSI reconstruction system.

Channel truth has two explicit simulator-side arguments: ``H_dl_true_for_ue``
is visible only to the UE encoder, and ``H_ul_true_for_channel`` is used only
to synthesize received samples. The BS receiver gets Y, pilot-only LS+U-Net
UL CSI, calibration statistics, and soft-symbol reliabilities.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

from models.twc_v1_k8_neural import K8PilotResidualUNet, K8SwinCSIModel


ROOT = Path(__file__).resolve().parents[1]


def _load(relative, name):
    path = ROOT / relative
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Conflicting module identity for {name}")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


# Import by file path so utils/__init__.py never initializes shared GPU policy.
link = _load("utils/twc_v1_k8_link.py", "_twc_v1_k8_system_link")
protocol = link.protocol
downlink = _load("utils/twc_v1_k8_downlink.py", "_twc_v1_k8_system_downlink")
calibration_module = _load("utils/twc_v1_k8_calibration.py", "_twc_v1_k8_system_calibration")
TWCv1K8CECalibration = calibration_module.TWCv1K8CECalibration


class K8FeedbackChain(torch.nn.Module):
    """Frozen CE plus a trainable UE/BS codec and the physical K8 feedback link."""

    def __init__(self, ce_model, calibration, codec_model):
        super().__init__()
        if not isinstance(ce_model, K8PilotResidualUNet):
            raise TypeError("ce_model must be the trained K8PilotResidualUNet")
        if not isinstance(calibration, TWCv1K8CECalibration):
            raise TypeError("The independent K8 CE calibration grid is required")
        if not isinstance(codec_model, K8SwinCSIModel):
            raise TypeError("codec_model must be a K8SwinCSIModel")
        self.ce_model = ce_model
        self.calibration = calibration
        self.codec_model = codec_model
        self.ce_model.eval()
        self.ce_model.requires_grad_(False)

    @classmethod
    def from_artifacts(cls, checkpoint_path, calibration_csv, *, device="cpu",
                       swin_kwargs=None):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format") != "twc_v1_k8_ce_checkpoint_v1":
            raise ValueError("Unexpected frozen CE checkpoint format")
        if int(checkpoint.get("step", -1)) != 5000:
            raise ValueError("Stage A CE checkpoint must be the completed 5000-update artifact")
        ce = K8PilotResidualUNet().to(device)
        ce.load_state_dict(checkpoint["model"], strict=True)
        ce.eval().requires_grad_(False)
        calibration = TWCv1K8CECalibration.from_csv(calibration_csv)
        codec = K8SwinCSIModel(**(swin_kwargs or {})).to(device)
        return cls(ce, calibration, codec)

    @staticmethod
    def _known_pilots(batch, device):
        pilots = torch.zeros((batch, protocol.K_USERS, protocol.N_SUBCARRIERS),
                             dtype=torch.complex64, device=device)
        for user, positions in enumerate(protocol.PILOT_POSITIONS_BY_USER):
            pilots[:, user, list(positions)] = 1.0 + 0.0j
        return pilots

    @staticmethod
    def _validate_channels(H_dl, H_ul):
        expected = (protocol.K_USERS, protocol.N_ANTENNAS, protocol.N_SUBCARRIERS)
        if (H_dl.dtype != torch.complex64 or H_ul.dtype != torch.complex64
                or H_dl.ndim != 4 or H_ul.ndim != 4
                or tuple(H_dl.shape[1:]) != expected or tuple(H_ul.shape[1:]) != expected
                or H_dl.shape != H_ul.shape or H_dl.shape[0] < 1
                or H_dl.device != H_ul.device
                or not bool(torch.isfinite(H_dl).all() & torch.isfinite(H_ul).all())):
            raise ValueError("Simulator channels must be finite complex64 [B,8,32,256] on one device")

    def forward(self, H_dl_true_for_ue, H_ul_true_for_channel, *,
                feedback_gamma_db, pilot_energy_per_ue,
                pilot_standard_noise, data_standard_noise,
                layout, energy_split=None):
        """Run the physical link and return a BS CSI estimate without truth input.

        ``layout`` is a HybridLayout or FDMALayout. For Hybrid, pass an
        EnergySplit whose pilot component equals ``pilot_energy_per_ue``.
        FDMA receives all remaining per-UE energy on its 24 orthogonal data RE.
        """
        self._validate_channels(H_dl_true_for_ue, H_ul_true_for_channel)
        H_dl = H_dl_true_for_ue
        H_ul = H_ul_true_for_channel
        batch, device = H_dl.shape[0], H_dl.device
        expected_noise = (batch, protocol.N_ANTENNAS, protocol.N_SUBCARRIERS)
        for value, name in ((pilot_standard_noise, "pilot_standard_noise"),
                            (data_standard_noise, "data_standard_noise")):
            if (value.dtype != torch.complex64 or tuple(value.shape) != expected_noise
                    or value.device != device or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} must be finite complex64 {expected_noise} on the channel device")
        n0 = protocol.feedback_noise_variance(float(feedback_gamma_db))
        if (float(pilot_energy_per_ue) not in self.calibration.energies
                or not self.calibration.gammas[0] <= float(feedback_gamma_db) <= self.calibration.gammas[-1]):
            raise ValueError("Pilot energy must be a calibrated node and gamma within calibrated range")
        ce_var = self.calibration.error_variance(float(feedback_gamma_db),
            float(pilot_energy_per_ue), device=device).expand(batch, protocol.K_USERS)

        pilot_y = link.transmit_pilots(H_ul, float(pilot_energy_per_ue),
            float(feedback_gamma_db), pilot_standard_noise)
        H_ls = link.estimate_ul_ls_interpolated(pilot_y, float(pilot_energy_per_ue))
        H_ul_est = self.ce_model(pilot_y, self._known_pilots(batch, device), H_ls,
            pilot_energy_per_ue=float(pilot_energy_per_ue), noise_variance=n0)

        if isinstance(layout, protocol.HybridLayout):
            if energy_split is None or not isinstance(energy_split, protocol.EnergySplit):
                raise TypeError("Hybrid links require an explicit K8 EnergySplit")
            if abs(energy_split.pilot - float(pilot_energy_per_ue)) > 1e-8:
                raise ValueError("Hybrid EnergySplit pilot energy differs from requested Ep")
            group_count = layout.group_count
            key_tx, shared_tx = self.codec_model.encode(H_dl, group_count=group_count)
            tx = link.transmit_feedback_data(H_ul, layout, energy_split,
                float(feedback_gamma_db), key_tx, shared_tx, data_standard_noise)
            recovered = link.recover_feedback_data(tx["Y"], H_ul_est, layout,
                energy_split, float(feedback_gamma_db), ce_var)
            Y_rx = pilot_y + tx["Y"]
            H_dl_hat = self.codec_model.decode_feedback(
                recovered["key_symbols_hat"], recovered["shared_symbols_hat"],
                group_count=group_count, Y_rx=Y_rx, H_ul_est=H_ul_est,
                layout=layout, feedback_gamma_db=float(feedback_gamma_db),
                noise_variance=n0, pilot_energy_per_ue=float(pilot_energy_per_ue),
                key_energy_per_ue=energy_split.key, shared_energy_per_ue=energy_split.shared,
                channel_error_variance=ce_var,
                key_reliability=recovered["key_reliability"],
                shared_reliability=recovered["shared_reliability"])
            soft = recovered
            energy = {"pilot": energy_split.pilot, "key": energy_split.key,
                      "shared": energy_split.shared,
                      "actual_data_per_ue": tx["actual_data_energy"]}
        elif isinstance(layout, protocol.FDMALayout):
            if energy_split is not None:
                raise ValueError("FDMA data energy is 64-Ep and has no key/shared split")
            group_count = 1
            data_energy = protocol.TOTAL_FRAME_ENERGY_PER_USER - float(pilot_energy_per_ue)
            data_tx = self.codec_model.encode_fdma(H_dl)
            tx = link.transmit_feedback_fdma_data(H_ul, layout, data_energy,
                float(feedback_gamma_db), data_tx, data_standard_noise)
            recovered = link.recover_feedback_fdma_data(tx["Y"], H_ul_est, layout,
                data_energy, float(feedback_gamma_db), ce_var)
            Y_rx = pilot_y + tx["Y"]
            packed = recovered["data_symbols_hat"]
            H_dl_hat = self.codec_model.decode_fdma_feedback(packed,
                Y_rx=Y_rx, H_ul_est=H_ul_est, layout=layout,
                feedback_gamma_db=float(feedback_gamma_db), noise_variance=n0,
                pilot_energy_per_ue=float(pilot_energy_per_ue),
                key_energy_per_ue=0.0, shared_energy_per_ue=data_energy,
                channel_error_variance=ce_var,
                reliability=recovered["reliability"])
            soft = recovered
            energy = {"pilot": float(pilot_energy_per_ue), "key": 0.0,
                      "shared": data_energy, "actual_data_per_ue": tx["actual_data_energy"]}
        else:
            raise TypeError("layout must be HybridLayout or FDMALayout")

        return {"H_dl_hat": H_dl_hat, "H_ul_est": H_ul_est, "H_ul_ls": H_ls,
            "Y_rx": Y_rx, "soft_symbols": soft,
            "channel_error_variance": ce_var, "energy": energy,
            "tx": tx, "n0": n0, "layout": layout,
            "group_count": group_count}

    @staticmethod
    def metrics(output, H_dl_true_for_metrics, *, rzf_alpha=0.01,
                downlink_snr_db=25.0):
        """Truth is accepted only by the metric helper, never by ``forward``'s BS."""
        nmse = downlink.downlink_nmse(output["H_dl_hat"], H_dl_true_for_metrics)
        precoder, power = downlink.design_rzf(output["H_dl_hat"], rzf_alpha)
        rates = downlink.evaluate_sum_rate(precoder, H_dl_true_for_metrics,
                                           downlink_snr_db)
        return {"dl_nmse_per_user": nmse, "precoder": precoder,
            "rzf_per_subcarrier_power": power, **rates}
