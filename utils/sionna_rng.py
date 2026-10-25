"""Opt-in RNG isolation for the unmodified Sionna PHY 2.0.1 Config.

No heavy imports at module import time. This module does not seed a trainer,
modify a channel generator, or implement checkpoint restoration. See
docs/SIONNA_RNG.md for the single-threaded/global-state contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.metadata
import pickle
import random
import threading


SUPPORTED_VERSION = "2.0.1"
PROTOCOL_VERSION = "sionna-rng-isolation-v1"
_LOCK = threading.RLock()


def validate_seed(seed):
    """Use a common explicit integer domain accepted by Python/NumPy/Torch."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an explicit integer in [0, 2**32-1], not None/bool")
    return seed


def _digest(value):
    # Locally generated state only: this utility never unpickles input files.
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class _DefaultState:
    def __init__(self, torch, np):
        self.torch, self.np = torch, np
        self.python = random.getstate()
        self.numpy = copy.deepcopy(np.random.get_state())
        # Materialize every visible CUDA RNG before Sionna's setter can reset it.
        self.cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        self.cpu = torch.get_rng_state().clone()

    def restore(self):
        torch, np = self.torch, self.np
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count != len(self.cuda):
            raise RuntimeError("Visible CUDA device set changed while RNG state was isolated")
        random.setstate(self.python)
        np.random.set_state(copy.deepcopy(self.numpy))
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)

    def fingerprint(self):
        return dict(python=_digest(self.python), numpy=_digest(self.numpy),
                    torch_cpu=_tensor_digest(self.cpu),
                    torch_cuda=[_tensor_digest(state) for state in self.cuda])


class _Sionna201State:
    """All version-sensitive private Config access is confined to this adapter."""

    @staticmethod
    def validate(torch, np, config):
        cls = type(config)
        if cls.__module__ != "sionna.phy.config" or cls.__name__ != "Config":
            raise RuntimeError("Expected the real sionna.phy.config.Config singleton")
        fields = ("_seed", "_py_rng", "_np_rng", "_torch_rngs", "_device", "_precision")
        if any(field not in vars(config) for field in fields):
            raise RuntimeError("Unsupported Sionna 2.0.1 private RNG layout")
        if config._seed is not None and type(config._seed) is not int:
            raise RuntimeError("Unsupported external Sionna seed type")
        if config._py_rng is not None and not isinstance(config._py_rng, random.Random):
            raise RuntimeError("Unsupported Sionna Python RNG")
        if config._np_rng is not None and not isinstance(config._np_rng, np.random.Generator):
            raise RuntimeError("Unsupported Sionna NumPy RNG")
        devices = ["cpu"] + ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                              if torch.cuda.is_available() else [])
        if (type(config._torch_rngs) is not dict or set(config._torch_rngs) != set(devices)
                or config.available_devices != devices or config.device not in devices):
            raise RuntimeError("Unsupported Sionna per-device Torch RNG registry")
        for device, generator in config._torch_rngs.items():
            if not isinstance(generator, torch.Generator) or str(generator.device) != device:
                raise RuntimeError(f"Unsupported Sionna Torch generator for {device}")
        if config.precision not in ("single", "double"):
            raise RuntimeError("Unsupported Sionna precision")

    def __init__(self, torch, np, config):
        self.validate(torch, np, config)
        self.config = config
        self.seed = config._seed
        self.device, self.precision = config._device, config._precision
        self.py_rng, self.np_rng = config._py_rng, config._np_rng
        self.py_state = None if self.py_rng is None else self.py_rng.getstate()
        self.np_state = None if self.np_rng is None else copy.deepcopy(self.np_rng.bit_generator.state)
        self.registry = config._torch_rngs
        self.generators = dict(self.registry)
        self.states = {device: rng.get_state().clone() for device, rng in self.generators.items()}

    def restore(self):
        config = self.config
        # Do NOT call config.seed on exit: its setter reseeds default Torch and
        # cannot recover advanced streams, original objects, or lazy None pools.
        if self.py_rng is not None:
            self.py_rng.setstate(self.py_state)
        if self.np_rng is not None:
            self.np_rng.bit_generator.state = copy.deepcopy(self.np_state)
        for device, generator in self.generators.items():
            generator.set_state(self.states[device])
        self.registry.clear()
        self.registry.update(self.generators)
        config._seed = self.seed
        config._py_rng, config._np_rng = self.py_rng, self.np_rng
        config._torch_rngs = self.registry
        config._device, config._precision = self.device, self.precision

    def fingerprint(self):
        return dict(python=None if self.py_rng is None else _digest(self.py_state),
                    numpy=None if self.np_rng is None else _digest(self.np_state),
                    torch={device: _tensor_digest(state) for device, state in self.states.items()})


def runtime():
    """Load genuine dependencies, rejecting other versions before heavy imports.

    Config's first import also invokes seed=None and can reseed default Torch.
    Preserve already-existing default streams around that import, too. CUDA
    context initialization itself cannot be undone, but its RNG state is saved.
    """
    with _LOCK:
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Sionna 2.0.1 is required; Sionna is not installed") from exc
        if version != SUPPORTED_VERSION:
            raise RuntimeError(f"Unsupported Sionna version {version}; require exactly {SUPPORTED_VERSION}")
        try:
            torch = importlib.import_module("torch")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise RuntimeError("Real PyTorch and NumPy dependencies are required") from exc
        defaults = _DefaultState(torch, np)
        try:
            config = importlib.import_module("sionna.phy").config
            _Sionna201State.validate(torch, np, config)
        except ImportError as exc:
            raise RuntimeError("Real Sionna 2.0.1 PHY runtime could not be imported") from exc
        finally:
            defaults.restore()
        return torch, np, config


class RNGSnapshot:
    """In-process state only, never a serialized training/checkpoint protocol."""

    def __init__(self, torch, np, config):
        self.defaults = _DefaultState(torch, np)
        self.sionna = _Sionna201State(torch, np, config)

    def restore(self):
        with _LOCK:
            try:
                self.sionna.restore()
            finally:
                self.defaults.restore()

    def fingerprint(self):
        return dict(seed=self.sionna.seed, private=self.sionna.fingerprint(),
                    defaults=self.defaults.fingerprint(), config_device=self.sionna.device,
                    config_precision=self.sionna.precision)


def snapshot_rng_state():
    """Capture seed, exact advanced states and original private RNG objects."""
    with _LOCK:
        return RNGSnapshot(*runtime())


@contextmanager
def isolated_sionna_seed(seed):
    """Temporarily seed real Sionna and defaults; restore on any kind of exit.

    Nested contexts are supported. All users of these global RNGs must remain
    on the same thread or cooperate with external synchronization. The lock
    serializes this API only; unrelated global RNG users are NOT protected.
    """
    seed = validate_seed(seed)
    with _LOCK:
        torch, np, config = runtime()
        saved = RNGSnapshot(torch, np, config)
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Official 2.0.1 semantics include per-device offsets and default
            # Torch reseeding. Do not override those offsets after the setter.
            config.seed = seed
            yield config
        finally:
            saved.restore()
