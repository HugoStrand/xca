"""Lazy CUDA backend loading, process device selection and collective error handling."""

from importlib import import_module

import triqs.utility.mpi as mpi

from ._backend_config import CUDA_ENABLED, CUDA_SMALL_ENABLED

_gpu_process_device = None


def block_sparse_evaluator_class(backend):
    if backend == 'cpu':
        from ._triqs_xca import BlockSparseDiagramEvaluator
        return BlockSparseDiagramEvaluator
    if backend != 'cuda':
        raise ValueError(f"Unknown block-sparse backend {backend!r}")
    if not CUDA_ENABLED:
        raise RuntimeError("Requested block-sparse backend 'cuda' is unavailable in this build")
    try:
        return getattr(import_module('._triqs_xca_cuda', __package__), 'BlockSparseCudaDiagramEvaluator')
    except (ImportError, AttributeError) as error:
        raise RuntimeError(f"Requested block-sparse backend 'cuda' is unavailable: {error}") from error


def dense_evaluator_class(backend):
    if backend == 'cpu':
        from ._triqs_xca import DenseDiagramEvaluator
        return DenseDiagramEvaluator
    if backend not in {'cuda', 'cuda_small'}:
        raise ValueError(f"Unknown dense backend {backend!r}")
    enabled = CUDA_SMALL_ENABLED if backend == 'cuda_small' else CUDA_ENABLED
    if not enabled:
        raise RuntimeError(f"Requested dense backend {backend!r} is unavailable in this build")
    class_name = 'DenseCudaSmallDiagramEvaluator' if backend == 'cuda_small' else 'DenseCudaDiagramEvaluator'
    try:
        return getattr(import_module('._triqs_xca_cuda', __package__), class_name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(f"Requested dense backend {backend!r} is unavailable: {error}") from error


def configure_device(requested):
    """Select one CUDA device for all XCA evaluators in this process."""
    global _gpu_process_device
    runtime = import_module('._triqs_xca_cuda', __package__)
    current = runtime.cuda_current_device()
    selected = current if requested is None else requested
    if selected < 0 or selected >= runtime.cuda_device_count():
        raise ValueError(f"CUDA device ordinal {selected} is unavailable")
    if _gpu_process_device is not None and (selected != _gpu_process_device or current != _gpu_process_device):
        raise RuntimeError("XCA GPU evaluators require one unchanged CUDA device per process")
    runtime.cuda_select_device(selected)
    _gpu_process_device = selected
    return selected


def collective_call(operation, phase="Native dense evaluation"):
    """Coordinate local device errors before callers enter result reductions."""
    result, error = None, None
    try:
        result = operation()
    except Exception as caught:
        error = caught
    failed = mpi.all_reduce(int(error is not None))
    if failed:
        if error is not None:
            raise RuntimeError(f"{phase} failed: {error}") from error
        raise RuntimeError(f"{phase} failed on another MPI rank")
    return result
