"""Bounded MPI selection, including empty ranks and dense paired ownership.

Runs in serial under CTest and can also be run with mpiexec -n 3. Numerical
references use the explicit index-vector API; partition checks never allocate
the full range, including for counts beyond int32.
"""
from unittest.mock import patch

import numpy as np
import triqs.utility.mpi as mpi
from triqs.operators import n

from api import make_model, make_solver, to_fock
from triqs_xca.solver import scatter_range_over_ranks


def check_partitions():
    for count in (0, 1, 2, 17, 2**40 + 7):
        for size in (1, 2, 3, 19):
            chunks = []
            for rank in range(size):
                with patch.object(mpi, 'size', size), patch.object(mpi, 'rank', rank):
                    chunks.append(scatter_range_over_ranks(count))
            assert chunks[0][0] == 0 and chunks[-1][1] == count
            assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
            lengths = [last - first for first, last in chunks]
            assert min(lengths) >= 0 and max(lengths) - min(lengths) <= 1
            assert lengths == sorted(lengths, reverse=True)
            assert all(isinstance(bound, int) for chunk in chunks for bound in chunk)


def check_evaluators():
    for use_blocks in (False, True):
        solver = make_solver(make_model('dimer'), use_blocks)
        evaluator, G = solver.d, solver.G0
        ops = [n('a', 0), n('a', 1)]
        for topology in ([[0, 1]], [[0, 2], [1, 3]]):
            topology = np.asarray(topology, dtype=np.int32)
            n_sigma = evaluator.get_num_self_energy_backbones(topology)
            bounds = scatter_range_over_ranks(n_sigma)
            full = to_fock(evaluator.compute_self_energy(G, topology, np.arange(n_sigma, dtype=np.int64)), solver.atom_diag).data
            local = to_fock(evaluator.compute_self_energy(G, topology, *bounds), solver.atom_diag).data
            np.testing.assert_allclose(mpi.all_reduce(local), full, rtol=5e-12, atol=1e-12)
            if not use_blocks:
                local_pairs = to_fock(evaluator.compute_self_energy_by_pairs(G, topology, *bounds), solver.atom_diag).data
                np.testing.assert_allclose(mpi.all_reduce(local_pairs), full, rtol=5e-12, atol=1e-12)
            n_corr = evaluator.get_num_single_ptcle_gf_backbones(topology)
            bounds = scatter_range_over_ranks(n_corr)
            indices = np.arange(n_corr, dtype=np.int64)
            full = evaluator.compute_single_ptcle_gf(G, topology, indices)
            local = evaluator.compute_single_ptcle_gf(G, topology, *bounds)
            np.testing.assert_allclose(mpi.all_reduce(local), full, rtol=5e-12, atol=1e-12)
            full = evaluator.compute_one_time_correlator(G, ops, ops, solver.atom_diag, topology, indices)
            local = evaluator.compute_one_time_correlator(G, ops, ops, solver.atom_diag, topology, *bounds)
            np.testing.assert_allclose(mpi.all_reduce(local), full, rtol=5e-12, atol=1e-12)


if __name__ == '__main__':
    check_partitions()
    check_evaluators()
