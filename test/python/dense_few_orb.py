"""FewOrb's Python constructors and topology API against the baseline evaluator.

Keep the fitted pole data explicit, so this test covers wrapping and evaluation
without depending on fitting accuracy. The existing MPI test covers Solver's
rank-local accumulation and final reductions using FewOrb as the dense default.
"""

import numpy as np

from triqs.atom_diag import AtomDiagReal, AtomDiagComplex
from triqs.operators import c, c_dag, n
from triqs_xca import DenseDiagramEvaluator, DenseDiagramEvaluatorFewOrb

from api import make_model, make_solver


def close(actual, expected):
    scale = max(1.0, np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11 * scale)


def evaluate(evaluator, G, ad, topology):
    densities = [n('a', 0), n('a', 1)]
    annihilation = [c('a', 0), c('a', 1)]
    creation = [c_dag('a', 0), c_dag('a', 1)]
    return (
        evaluator.compute_self_energy(G_ppsc=G, topology=topology),
        evaluator.compute_single_ptcle_gf(G_ppsc=G, topology=topology),
        evaluator.compute_one_time_correlator(G_ppsc=G, ops_tau=densities, ops_0=densities, ad=ad, topology=topology),
        evaluator.compute_one_time_correlator(G_ppsc=G, ops_tau=annihilation, ops_0=creation, ad=ad, topology=topology),
    )


def arrays(results):
    sigma, *correlators = results
    return [block.data for _, block in sigma] + correlators


def check_wrapping(complex_hamiltonian, dynamic_interactions):
    model = make_model('dimer')
    if not complex_hamiltonian:
        model['H'] = (0.3 * (n('a', 0) + n('a', 1)) + 1.5 * n('a', 0) * n('a', 1)
                      + 0.4 * (c_dag('a', 0) * c('a', 1) + c_dag('a', 1) * c('a', 0)))
    solver = make_solver(model, use_blocks=False)
    assert isinstance(solver.d, DenseDiagramEvaluatorFewOrb)
    assert isinstance(solver.atom_diag, AtomDiagComplex if complex_hamiltonian else AtomDiagReal)

    args = (solver.hyb.poles, solver.hyb.coefficients, solver.mesh_tau, solver.atom_diag)
    if dynamic_interactions:
        # The dynamical interaction uses the same fitted poles as hybridization.
        ops = [n('a', 0), n('a', 1)]
        coeffs = np.array([[[0.08, 0.01j], [-0.01j, 0.04]],
                           [[0.05, -0.02j], [0.02j, 0.07]],
                           [[0.03, 0.01], [0.01, 0.06]]], dtype=complex)
        args += (ops, coeffs)

    baseline = DenseDiagramEvaluator(*args)
    few_orb = DenseDiagramEvaluatorFewOrb(*args)
    uncached = DenseDiagramEvaluatorFewOrb(*args, cache_bytes=0)
    for hidden in ('action', 'edge_cache_statistics'):
        assert not hasattr(few_orb, hidden), hidden
    G = solver.G0.copy()
    topologies = (np.array([[0, 1]], dtype=np.int32),
                  np.array([[0, 2], [1, 3]], dtype=np.int32))

    retained = []
    for topology in topologies:
        reference = arrays(evaluate(baseline, G, solver.atom_diag, topology))
        assert all(np.max(np.abs(value)) > 1e-10 for value in reference)
        for evaluator in (few_orb, uncached):
            result = evaluate(evaluator, G, solver.atom_diag, topology)
            for actual, expected in zip(arrays(result), reference):
                close(actual, expected)
            retained.append((result, [value.copy() for value in arrays(result)]))

    # Reusing the same G allocation must invalidate numerical cache contents.
    for _, block in G:
        block.data[:] *= 0.81
    topology = topologies[-1]
    changed_reference = arrays(evaluate(baseline, G, solver.atom_diag, topology))
    for evaluator in (few_orb, uncached):
        for _ in range(2):
            result = evaluate(evaluator, G, solver.atom_diag, topology)
            for actual, expected in zip(arrays(result), changed_reference):
                close(actual, expected)
    for result, snapshot in retained:
        for actual, expected in zip(arrays(result), snapshot):
            close(actual, expected)


if __name__ == '__main__':
    for complex_hamiltonian in (False, True):
        for dynamic_interactions in (False, True):
            check_wrapping(complex_hamiltonian, dynamic_interactions)
