"""Select one JAX backend for physics, perception, policy and optimizer."""
from dataclasses import replace
import os
# Share the display GPU without reserving 75% of VRAM at process startup.
# Respect an explicit user setting; XLA reads this before backend initialization.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax


def configure_device(cfg):
    requested = cfg.device
    if requested not in ('auto', 'cpu', 'gpu'):
        raise ValueError('device must be auto, cpu or gpu')
    if requested == 'auto':
        try:
            devices = jax.devices('gpu')
        except RuntimeError:
            devices = jax.devices('cpu')
    else:
        try:
            devices = jax.devices(requested)
        except RuntimeError as exc:
            raise RuntimeError(
                f'Requested {requested} is unavailable. Activate proyecto_dron, check '
                'nvidia-smi and remove a conflicting JAX_PLATFORMS setting; '
                'use --device cpu or auto for CPU fallback.') from exc
    device = devices[0]
    # Must run before creating RNGs/weights/scenarios; Crazyflow also receives
    # this resolved platform. Explicit CPU stays on CPU even on a GPU host.
    jax.config.update('jax_default_device', device)
    return replace(cfg, device=device.platform), device


def device_info(device):
    return {'platform': device.platform, 'kind': device.device_kind,
            'id': device.id, 'jax_version': jax.__version__,
            'jax_platforms': os.environ.get('JAX_PLATFORMS'),
            'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}
