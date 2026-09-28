"""Read-only physical-link probes used only by diagnose_low_feedback_snr.

No model/source edits. Wrappers call each original operator exactly once, return
its unchanged output, draw no random numbers, and are removed on context exit.
No neural representation is interpreted as a post-detection SNR.
"""
from contextlib import ExitStack
import importlib
import math
from unittest.mock import patch


class FeedbackLinkProbe:
    def __init__(self, methods, batch_state):
        self.methods = methods
        self.batch_state = batch_state
        self.rows = []
        self.context = None
        self.expected_as = None
        self.individual_as_power = None
        self.baseline_capture = None
        self.stack = ExitStack()

    def __enter__(self):
        try:
            for method in self.methods:
                model = method.model
                handle = model.register_forward_pre_hook(self._before(method))
                self.stack.callback(handle.remove)
                if method.model_key == 'proposed':
                    original_noise, original_tx = model.add_noise, model.transmit
                    self.stack.enter_context(patch.object(model, 'add_noise', self._proposal_noise(original_noise)))
                    self.stack.enter_context(patch.object(model, 'transmit', self._transmit(original_tx)))
                else:
                    module = importlib.import_module(type(model).__module__)
                    self.stack.enter_context(patch.object(
                        module, 'fdma_feedback_mrc', self._mrc(module.fdma_feedback_mrc),
                    ))
            from models import baseline_common
            self.stack.enter_context(patch.object(
                baseline_common, 'add_relative_awgn',
                self._baseline_noise(baseline_common.add_relative_awgn),
            ))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def _before(self, method):
        def hook(model, inputs):
            if model.training:
                raise RuntimeError('Diagnostics require eval mode; running power must stay frozen.')
            dl, ul, snr = inputs[:3]
            state = self.batch_state
            if dl.data_ptr() != state['H_dl'].data_ptr() or ul.data_ptr() != state['H_ul'].data_ptr():
                raise RuntimeError('Methods are not receiving the same cached channel tensors.')
            df, da = method.allocations[0]
            cfg_index = None
            if method.model_key == 'proposed':
                cfg_index = list(map(tuple, method.config.GAMMA_GRID)).index((df, da))
                if tuple(inputs[3:6]) != (df, da, cfg_index):
                    raise RuntimeError('Allocation or running-power index was changed.')
            self.context = dict(
                method_key=method.method_key, batch_index=state['batch_index'],
                first_sample_index=state['batch_index'] * dl.shape[0],
                batch_size=dl.shape[0], allocation_df=df, allocation_da=da,
                cfg_index=cfg_index, feedback_snr_db=float(snr), downlink_snr_db=25,
            )
            self.expected_as = self.individual_as_power = self.baseline_capture = None
        return hook

    def _transmit(self, original):
        def wrapped(zf, za, h, snr, df, da):
            if da:
                terms = h[:, :, :, h.shape[1]*df:h.shape[1]*df+da] * za.unsqueeze(2)
                self.expected_as = terms.sum(dim=1)
                self.individual_as_power = float(terms.abs().square().sum(dim=1).mean())
            return original(zf, za, h, snr, df, da)
        return wrapped

    def _record(self, clean, noisy, snr, branch):
        import torch
        if self.context is None or float(snr) != self.context['feedback_snr_db']:
            raise RuntimeError('Physical probe has no matching forward context.')
        noise = noisy - clean
        signal_power = float(clean.abs().square().mean())
        variance = signal_power * 10.0 ** (-float(snr) / 10.0)
        noise_power = float(noise.abs().square().mean())
        row = dict(
            **self.context, branch=branch,
            measurement_location='BS antenna samples before MRC/DFT/neural decoder',
            power_average_axes='all elements of this active stream and channel batch',
            received_shape=str(tuple(clean.shape)),
            clean_received_power=signal_power,
            expected_complex_noise_variance=variance,
            expected_noise_variance_per_real_component=variance / 2,
            realized_noise_power=noise_power,
            realized_noise_real_power=float(noise.real.square().mean()),
            realized_noise_imag_power=float(noise.imag.square().mean()),
            measured_pre_receiver_snr_db=(
                10 * math.log10(signal_power / noise_power)
                if signal_power > 0 and noise_power > 0 else None
            ),
        )
        if branch == 'AS':
            if self.expected_as is None or not torch.allclose(clean, self.expected_as, rtol=1e-5, atol=1e-7):
                raise RuntimeError('AS noise reference is not the complete coherent multi-user sum.')
            row.update(
                as_full_sum_verified=True,
                as_sum_individual_received_powers=self.individual_as_power,
                as_coherent_cross_term_power=signal_power-self.individual_as_power,
            )
        self.rows.append(row)
        return row, noise

    def _proposal_noise(self, original):
        def wrapped(clean, snr):
            noisy = original(clean, snr)
            branch = {4: 'FDMA', 3: 'AS'}.get(clean.ndim)
            if branch is None:
                raise RuntimeError('Unexpected physical observation shape.')
            self._record(clean, noisy, snr, branch)
            return noisy
        return wrapped

    def _baseline_noise(self, original):
        def wrapped(clean, snr):
            noisy = original(clean, snr)
            row, noise = self._record(clean, noisy, snr, 'FDMA')
            self.baseline_capture = (clean, noise, row)
            return noisy
        return wrapped

    def _mrc(self, original):
        def wrapped(codewords, h, snr, df, H_ul_est=None):
            import torch
            from models.baseline_common import normalize_complex_symbols
            output = original(codewords, h, snr, df, H_ul_est=H_ul_est)
            if self.baseline_capture is None:
                raise RuntimeError('MRC diagnostics did not capture the actual channel noise.')
            clean, noise, row = self.baseline_capture
            estimate_h = h if H_ul_est is None else H_ul_est
            segments = torch.stack([
                estimate_h[:, k, :, k*df:(k+1)*df] for k in range(h.shape[1])
            ], dim=1)
            energy = segments.abs().square().sum(dim=2)
            denominator = energy.clamp_min(1e-8)
            projected_noise = (segments.conj() * noise).sum(dim=2) / denominator
            clean_mrc = (segments.conj() * clean).sum(dim=2) / denominator
            symbols = normalize_complex_symbols(torch.complex(codewords[..., :df], codewords[..., df:]))
            estimated = torch.complex(output[..., :df], output[..., df:])
            symbol_power = float(symbols.abs().square().mean())
            error_power = float((estimated-symbols).abs().square().mean())
            row.update(
                mrc_perfect_csi=H_ul_est is None or torch.equal(H_ul_est, h),
                mrc_expected_complex_noise_variance=float((
                    row['expected_complex_noise_variance'] * energy / denominator.square()
                ).mean()),
                mrc_realized_noise_power=float(projected_noise.abs().square().mean()),
                mrc_clean_symbol_error_power=float((clean_mrc-symbols).abs().square().mean()),
                mrc_symbol_power=symbol_power, mrc_symbol_error_power=error_power,
                mrc_symbol_nmse_linear=error_power / symbol_power if symbol_power > 0 else None,
            )
            return output
        return wrapped

    def verify_coverage(self, num_batches, snrs):
        expected = set()
        for method in self.methods:
            df, da = method.allocations[0]
            for branch, active in [('FDMA', df), ('AS', da)]:
                if active:
                    expected.update((method.method_key, b, float(s), branch)
                                    for b in range(num_batches) for s in snrs)
        keys = [(r['method_key'], r['batch_index'], r['feedback_snr_db'], r['branch']) for r in self.rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise RuntimeError('Missing/duplicate physical-link diagnostic rows.')
