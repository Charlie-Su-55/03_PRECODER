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
        self.installation = None
        self.cleanup_report = None

    def __enter__(self):
        self.installation = ProbeInstallationSnapshot(self.methods)
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
            self.cleanup_report = self.installation.report()
            raise

    def __exit__(self, *exc):
        try:
            return self.stack.__exit__(*exc)
        finally:
            self.cleanup_report = self.installation.report()
            self.expected_as = self.baseline_capture = None

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


# Runtime snapshots are diagnostic-only. Tensor copies below live on CPU;
# all three replay trials use one existing set of GPU model objects.
def tensor_record(value):
    import hashlib
    import torch
    cpu = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                requires_grad=value.requires_grad, sha256=hashlib.sha256(raw).hexdigest())


def changed_records(before, after):
    return [dict(name=k, before=before.get(k), after=after.get(k))
            for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)]


def callable_identity(value):
    # Reading a bound method creates a fresh wrapper, so compare self + function.
    return (id(getattr(value, '__self__', None)), id(getattr(value, '__func__', value)))


class ProbeInstallationSnapshot:
    """Observe all module hooks and the exact attributes the probe wraps."""
    def __init__(self, methods):
        self.methods = methods
        self.targets = []
        seen = set()
        for method in methods:
            if method.model_key == 'proposed':
                targets = [(method.model, 'add_noise'), (method.model, 'transmit')]
            else:
                targets = [(importlib.import_module(type(method.model).__module__), 'fdma_feedback_mrc')]
            for owner, attr in targets:
                if (id(owner), attr) not in seen:
                    self.targets.append((method.method_key+'.'+attr, owner, attr))
                    seen.add((id(owner), attr))
        # FeedbackLinkProbe patches this function even for a proposal-only list.
        module = importlib.import_module('models.baseline_common')
        self.targets.append(('baseline_common.add_relative_awgn', module, 'add_relative_awgn'))
        self.before = self.capture()

    def capture(self):
        functions, hooks = {}, {}
        for name, owner, attr in self.targets:
            functions[name] = dict(
                identity=callable_identity(getattr(owner, attr)),
                instance_attribute=attr in vars(owner),
            )
        for method in self.methods:
            for name, module in method.model.named_modules():
                for attr, entries in vars(module).items():
                    if 'hook' in attr and isinstance(entries, dict):
                        hooks[f'{method.method_key}.{name}.{attr}'] = [
                            (str(k), callable_identity(v) if callable(v) else repr(v))
                            for k, v in entries.items()
                        ]
        return dict(functions=functions, hooks=hooks)

    def report(self):
        after = self.capture()
        result = {k: changed_records(self.before[k], after[k]) for k in self.before}
        return dict(passed=not any(result.values()), **result)


def _payload(value):
    """CPU snapshot for tensor/primitive/container caches; flag opaque objects."""
    import copy
    import random
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return dict(kind='tensor', value=value.detach().cpu().clone(), token=tensor_record(value))
    if isinstance(value, (dict, list, tuple)):
        pairs = value.items() if isinstance(value, dict) else enumerate(value)
        children = {k: _payload(v) for k, v in pairs}
        return dict(kind=type(value).__name__, children=children,
                    token={str(k): v['token'] for k, v in children.items()})
    if value is None or isinstance(value, (str, int, float, bool)):
        return dict(kind='scalar', value=copy.deepcopy(value), token=repr(value))
    if isinstance(value, (torch.Generator, np.random.Generator, np.random.RandomState, random.Random)):
        return dict(kind='generator_ref', value=value, token=f'{type(value).__name__}:{id(value)}')
    return dict(kind='opaque', value=value, token=f'UNSUPPORTED {type(value).__name__}: {value!r}')


def _restore_payload(payload):
    if payload['kind'] == 'tensor':
        return payload['value'].to(payload['token']['device']).clone().requires_grad_(payload['token']['requires_grad'])
    if 'children' in payload:
        children = {k: _restore_payload(v) for k, v in payload['children'].items()}
        if payload['kind'] == 'dict':
            return children
        values = [children[k] for k in sorted(children)]
        return tuple(values) if payload['kind'] == 'tuple' else values
    return payload['value']


def _opaque_payload_paths(payload, name):
    if payload['kind'] == 'opaque':
        return [name]
    return [path for key, child in payload.get('children', {}).items()
            for path in _opaque_payload_paths(child, name+'.'+str(key))]


class ModelMemorySnapshot:
    """Include non-persistent buffers, which state_dict() intentionally omits."""
    def __init__(self, methods, channels):
        import torch
        self.methods, self.channels = methods, channels
        self.roots = {m.method_key: m.model for m in methods}
        self.modules, self.registries, self.values, self.caches = {}, {}, {}, {}
        self.input_copies = [x.detach().cpu().clone() for x in channels]
        for method in methods:
            for name, module in method.model.named_modules():
                key = method.method_key+'.'+name
                self.modules[key] = module
                self.registries[key] = (
                    dict(module._parameters), dict(module._buffers), dict(module._modules),
                    set(module._non_persistent_buffers_set), module.training,
                )
                for attr in ('_parameters', '_buffers'):
                    for field, value in getattr(module, attr).items():
                        self.values[key, attr, field] = None if value is None else value.detach().cpu().clone()
                self.caches[key] = {k: _payload(v) for k, v in self._cache_items(module).items()}
        self.before = self.capture()

    @staticmethod
    def _cache_items(module):
        import random
        import numpy as np
        import torch
        return {k: v for k, v in vars(module).items()
                if torch.is_tensor(v)
                or isinstance(v, (torch.Generator, np.random.Generator, np.random.RandomState, random.Random))
                or ('cache' in k.lower() and k not in ('_parameters', '_buffers', '_modules'))}

    def capture(self):
        result = dict(parameters={}, buffers={}, modes={}, module_tree={}, extra_caches={},
                      unsupported_caches={}, inputs={})
        for method in self.methods:
            for name, module in method.model.named_modules():
                key = method.method_key+'.'+name
                result['module_tree'][key] = id(module)
                result['modes'][key] = module.training
                for attr, group in (('_parameters', 'parameters'), ('_buffers', 'buffers')):
                    for field, value in getattr(module, attr).items():
                        token = None if value is None else tensor_record(value)
                        result[group][key+'.'+field] = dict(
                            tensor=token, persistent=field not in module._non_persistent_buffers_set,
                            object_id=id(value) if value is not None else None,
                        )
                for field, value in self._cache_items(module).items():
                    payload = _payload(value)
                    result['extra_caches'][key+'.'+field] = payload['token']
                    for path in _opaque_payload_paths(payload, key+'.'+field):
                        result['unsupported_caches'][path] = 'Opaque cache cannot be verified/restored safely'
        for i, value in enumerate(self.channels):
            result['inputs'][('H_dl', 'H_ul')[i]] = tensor_record(value)
        return result

    def report(self):
        current = self.capture()
        changes = {k: changed_records(self.before[k], current[k]) for k in self.before}
        unsupported = sorted(set(self.before['unsupported_caches']) | set(current['unsupported_caches']))
        return dict(passed=not any(changes.values()) and not unsupported,
                    coverage_complete=not unsupported, unverified_cache_paths=unsupported, **changes)

    def restore(self):
        import torch
        # Report is always captured BEFORE calling restore; restoring is not a PASS.
        with torch.no_grad():
            for method in self.methods:
                method.model = self.roots[method.method_key]
            for key, module in self.modules.items():
                params, buffers, children, nonpersistent, training = self.registries[key]
                module._parameters.clear()
                module._parameters.update(params)
                module._buffers.clear()
                module._buffers.update(buffers)
                module._modules.clear()
                module._modules.update(children)
                module._non_persistent_buffers_set = set(nonpersistent)
                module.training = training
                for attr in ('_parameters', '_buffers'):
                    for field, value in getattr(module, attr).items():
                        if value is not None:
                            target = self.values[key, attr, field]
                            # Avoid gratuitous copy_/version changes for unchanged tensors.
                            token = self.before['parameters' if attr == '_parameters' else 'buffers'][key+'.'+field]['tensor']
                            if tensor_record(value) != token:
                                value.data = target.to(token['device']).clone()
                                value.requires_grad_(token['requires_grad'])
                for field in set(self._cache_items(module)) - set(self.caches[key]):
                    delattr(module, field)
                for field, payload in self.caches[key].items():
                    if field not in vars(module) or _payload(vars(module)[field])['token'] != payload['token']:
                        setattr(module, field, _restore_payload(payload))
            for value, saved in zip(self.channels, self.input_copies):
                if not torch.equal(value.cpu(), saved):
                    value.copy_(saved.to(value.device))
        return self.report()


def discover_independent_generators(methods, evaluator):
    """Inspect active model attributes / evaluator globals, not unused Sionna code."""
    import random
    import numpy as np
    import torch
    found = {}
    def visit(prefix, value, depth=0):
        if isinstance(value, (torch.Generator, np.random.Generator, np.random.RandomState, random.Random)):
            found[prefix] = value
        elif depth < 3 and isinstance(value, (dict, list, tuple)):
            pairs = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in pairs:
                visit(prefix+'.'+str(key), child, depth+1)
    for method in methods:
        for name, module in method.model.named_modules():
            visit(method.method_key+'.'+name, vars(module))
    visit('evaluator', vars(evaluator))
    return found


class RNGSnapshot:
    def __init__(self, methods, evaluator):
        import copy
        import random
        import numpy as np
        import torch
        self.methods, self.evaluator = methods, evaluator
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.cpu = torch.get_rng_state().clone()
        self.cuda = [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        self.independent = discover_independent_generators(methods, evaluator)
        self.independent_states = {}
        for key, obj in self.independent.items():
            if isinstance(obj, torch.Generator):
                state = obj.get_state().clone()
            elif isinstance(obj, np.random.Generator):
                state = copy.deepcopy(obj.bit_generator.state)
            else:
                state = obj.get_state() if isinstance(obj, np.random.RandomState) else obj.getstate()
            self.independent_states[key] = state

    def fingerprints(self):
        import hashlib
        import pickle
        import torch
        def digest(value):
            if torch.is_tensor(value):
                return tensor_record(value)['sha256']
            return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
        return dict(python=digest(self.python), numpy=digest(self.numpy), torch_cpu=digest(self.cpu),
                    cuda={str(i): digest(x) for i, x in enumerate(self.cuda)},
                    independent={k: digest(v) for k, v in self.independent_states.items()})

    def restore(self):
        import copy
        import random
        import numpy as np
        import torch
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)
        for key, obj in self.independent.items():
            state = self.independent_states[key]
            if isinstance(obj, torch.Generator):
                obj.set_state(state)
            elif isinstance(obj, np.random.Generator):
                obj.bit_generator.state = copy.deepcopy(state)
            elif isinstance(obj, np.random.RandomState):
                obj.set_state(state)
            else:
                obj.setstate(state)
        return RNGSnapshot(self.methods, self.evaluator).fingerprints() == self.fingerprints()


def rng_comparison(left, right):
    if left is None or right is None:
        return dict(passed=False, error='Missing RNG snapshot')
    equality = {key: left.get(key) == right.get(key)
                for key in ('python', 'numpy', 'torch_cpu', 'cuda', 'independent')}
    return dict(passed=all(equality.values()), equal=equality, left=left, right=right)


def numerical_backend_settings():
    import os
    import torch
    return dict(
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_version=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        sdpa_math=torch.backends.cuda.math_sdp_enabled(),
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        diagnostic_backend_override=False,
    )
