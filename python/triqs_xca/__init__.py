from ._triqs_xca import DenseDiagramEvaluator, BlockSparseDiagramEvaluator
from ._triqs_xca import sigma_nca, sigma_oca, expectation_value, convolve_ppsc, trace
from .solver import Solver


def __getattr__(name):
    # GPU extensions are optional; importing the CPU package never loads them.
    if name in {'DenseCudaDiagramEvaluator', 'DenseCudaSmallDiagramEvaluator'}:
        from .gpu import dense_evaluator_class
        return dense_evaluator_class('cuda_small' if 'Small' in name else 'cuda')
    raise AttributeError(name)
