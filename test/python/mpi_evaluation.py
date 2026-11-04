"""Topology-local evaluators and complete Solver sums, in serial and under MPI.

C++ mpi_evaluator checks local ownership against prepared Backbone references.
Here we check the normal Python import/initialization route, binding surface,
Solver signs and accumulation, and the number of final reduction calls.
"""

from triqs_xca import DenseDiagramEvaluatorFewOrb, BlockSparseDiagramEvaluator
import triqs.utility.mpi as mpi
from triqs.operators import n
import numpy as np
import os
from unittest.mock import patch

from api import make_model, make_solver
from triqs_xca.diag import all_connected_pairings


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-12)


def reduce_blocks(value):
    return [mpi.all_reduce(block.data) for _, block in value]


def check_blocks(value, expected):
    for (_, block), reference in zip(value, expected):
        close(block.data, reference)


def check_solver(use_blocks):
    solver = make_solver(make_model('dimer'), use_blocks)
    evaluator, G = solver.d, solver.G0
    assert isinstance(evaluator, BlockSparseDiagramEvaluator if use_blocks else DenseDiagramEvaluatorFewOrb)
    for removed in ('reset', 'get_num_self_energy_backbones',
                    'get_num_single_ptcle_gf_backbones',
                    'print_self_energy_backbone', 'print_single_ptcle_gf_backbone',
                    'compute_self_energy_by_pairs', 'compute_self_energy_single_direction',
                    'compute_self_energy_both_directions'):
        assert not hasattr(evaluator, removed), removed
    assert not hasattr(solver, 'eval_pseudo_particle_self_energy_order_by_pairs')

    ops = [n('a', 0), n('a', 1)]
    first_order = np.array([[0, 1]], dtype=np.int32)
    local_sigma = evaluator.compute_self_energy(G, first_order)
    local_spgf = evaluator.compute_single_ptcle_gf(G, first_order)
    # Dense evaluation shares the two endpoint directions in one task;
    # block-sparse evaluation retains one task for each direction.
    first_order_tasks = 2 if use_blocks else 1
    if mpi.rank >= first_order_tasks:
        assert all(np.count_nonzero(block.data) == 0 for _, block in local_sigma)
    if mpi.rank > 0:
        assert np.count_nonzero(local_spgf) == 0
    if not use_blocks:
        assert (np.max(np.abs(local_sigma[0].data)) > 1e-10) == (mpi.rank == 0)

    # Public Solver topology helpers continue to return complete results.
    expected_sigma = reduce_blocks(local_sigma)
    expected_spgf = mpi.all_reduce(local_spgf)
    with patch.object(mpi, 'all_reduce', wraps=mpi.all_reduce) as reduce:
        check_blocks(solver.eval_pseudo_particle_self_energy_topology(G, first_order), expected_sigma)
        assert reduce.call_count == len(expected_sigma)
    with patch.object(mpi, 'all_reduce', wraps=mpi.all_reduce) as reduce:
        close(solver.eval_single_particle_greens_function_topology(G, first_order).data, expected_spgf)
        assert reduce.call_count == 1

    # References accumulate local topology results explicitly, independent of
    # Solver's order loops. Order three supplies several connected topologies.
    max_order = 3
    sigma = solver.get_zero_pseudo_particle_propagator()
    spgf = np.zeros_like(local_spgf)
    corr = np.zeros((len(solver.mesh_tau), len(ops), len(ops)), dtype=complex)
    for order in range(1, max_order + 1):
        order_sigma = solver.get_zero_pseudo_particle_propagator()
        for _, topology in all_connected_pairings(order):
            topology = np.asarray(topology, dtype=np.int32)
            order_sigma -= evaluator.compute_self_energy(G, topology)
            spgf += (-1)**order * evaluator.compute_single_ptcle_gf(G, topology)
            corr += (-1)**order * evaluator.compute_one_time_correlator(G, ops, ops, solver.atom_diag, topology)
        sigma += order_sigma
    expected_sigma = reduce_blocks(sigma)
    expected_spgf, expected_corr = mpi.all_reduce(spgf), mpi.all_reduce(corr)

    with patch.object(mpi, 'all_reduce', wraps=mpi.all_reduce) as reduce:
        check_blocks(solver.eval_pseudo_particle_self_energy(G, max_order), expected_sigma)
        assert reduce.call_count == len(expected_sigma)
    with patch.object(mpi, 'all_reduce', wraps=mpi.all_reduce) as reduce:
        close(solver.eval_single_particle_greens_function(G, max_order).data, expected_spgf)
        assert reduce.call_count == 1
    with patch.object(mpi, 'all_reduce', wraps=mpi.all_reduce) as reduce:
        close(solver.eval_one_time_correlator(G, max_order, ops, ops).data, expected_corr)
        assert reduce.call_count == 1


if __name__ == '__main__':
    assert mpi.size == int(os.environ.get('XCA_TEST_MPI_SIZE', '1'))
    for use_blocks in (False, True):
        check_solver(use_blocks)
