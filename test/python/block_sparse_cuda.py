import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from h5 import HDFArchive
from triqs.operators import n, c, c_dag
from triqs.gfs import Gf
import triqs.utility.mpi as mpi
from triqs_xca import Solver, gpu
from triqs_xca._backend_config import CUDA_ENABLED


def make_solver(backend='cpu', partition='number', dynamic=False):
    Q = sum(n('f', i) for i in range(3))
    H = sum((.13+.07*i)*n('f', i) for i in range(3))
    H += sum(.2*n('f', i)*n('f', j) for i in range(3) for j in range(i))
    if partition != 'occupation':
        for i in range(3):
            for j in range(i):
                if partition == 'number' or i == 1:
                    z = .03*(i+j+1) + .017j*(i-j)
                    H += z*c_dag('f', i)*c('f', j) + z.conjugate()*c_dag('f', j)*c('f', i)
    sym = {'occupation': 'automatic', 'number': [Q],
           'mixed': [n('f', 0)+n('f', 1), n('f', 2)]}[partition]
    S = Solver(H, 2., 20., 1e-10, [['f', 3]], conserved_operators=sym,
               verbose=False, block_sparse_backend=backend)
    coeff = np.zeros((2, 3, 3), complex)
    for p in range(2):
        coeff[p] = np.eye(3)*(.3-.1*p)
        if partition == 'number':
            for i in range(3):
                for j in range(3):
                    if i != j: coeff[p, i, j] = .015*(1+i+j) + .007j*(i-j)
    S.set_hybridization_poles_and_coefficients(np.array([1.3, -.8]), coeff)
    if dynamic:
        S.has_dynamic_interactions = True
        S.dynint_ops = [Q, c_dag('f', 0)*c('f', 1)]
        S.dynint_coeffs = np.array([[[.07, 0.], [0., 0.]], [[0., 0.], [0., .03+.01j]]], complex)
    return S


class BlockSparseCudaTests(unittest.TestCase):
    def test_preferences_validation_and_archives(self):
        for kwargs in ({'block_sparse_backend': 'typo'},
                       {'block_sparse_backend': 'cuda', 'conserved_operators': []},
                       {'block_sparse_workspace_budget': 0}, {'block_sparse_operator_budget': True}):
            with self.assertRaises(ValueError):
                Solver(n('f', 0), 2., 10., 1e-10, [['f', 1]], verbose=False, **kwargs)
        S = make_solver('cuda')
        S.block_sparse_operator_budget = 64*1024*1024
        clone = copy.deepcopy(S)
        self.assertEqual(clone.block_sparse_backend, 'cuda')
        self.assertEqual(clone.block_sparse_operator_budget, S.block_sparse_operator_budget)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'solver.h5')
            with HDFArchive(path, 'w') as archive: archive['solver'] = S
            with HDFArchive(path, 'r') as archive: restored = archive['solver']
            self.assertEqual(restored.block_sparse_backend, 'cuda')
            self.assertEqual(restored.block_sparse_workspace_budget, S.block_sparse_workspace_budget)
            self.assertFalse(hasattr(restored, 'd'))
        data = S.__reduce_to_dict__()
        for key in ('block_sparse_backend', 'block_sparse_workspace_budget', 'block_sparse_operator_budget'):
            data.pop(key)
        restored = Solver.__factory_from_dict__('Solver', data)
        self.assertEqual(restored.block_sparse_backend, 'cpu')
        if not CUDA_ENABLED:
            with self.assertRaisesRegex(RuntimeError, 'unavailable'): S.init_diagram_evaluator()

    @unittest.skipUnless(CUDA_ENABLED, 'requires CUDA TRIQS')
    def test_registration_and_all_observables(self):
        import triqs_xca
        from triqs_xca import _triqs_xca_cuda as module
        self.assertIs(triqs_xca.BlockSparseCudaDiagramEvaluator, module.BlockSparseCudaDiagramEvaluator)
        self.assertEqual(module.BlockSparseCudaDiagramEvaluator.__module__, 'triqs_xca._triqs_xca_cuda')
        for partition in ('occupation', 'number', 'mixed'):
            for dynamic in (False, True):
                with self.subTest(partition=partition, dynamic=dynamic):
                    expected, actual = make_solver('cpu', partition, dynamic), make_solver('cuda', partition, dynamic)
                    expected.init_diagram_evaluator(); actual.init_diagram_evaluator()
                    self.assertTrue(actual._uses_gpu_evaluator)
                    self.assertFalse(actual.use_dense_solver)
                    for t in (np.array([[0, 1]], np.int32), np.array([[0, 2], [1, 3]], np.int32)):
                        se = actual._Solver__eval_pseudo_particle_self_energy_topology_loop(actual.G0, t)
                        reference = expected.d.compute_self_energy(expected.G0, t)
                        for b in range(len(reference)):
                            np.testing.assert_allclose(se[b].data, reference[b].data, rtol=1e-11, atol=1e-13)
                        spgf = actual._Solver__eval_single_particle_greens_function_topology_loop(actual.G0, t)
                        np.testing.assert_allclose(spgf.data, expected.d.compute_single_ptcle_gf(expected.G0, t), rtol=1e-11, atol=1e-13)
                        mu = [sum(n('f', i) for i in range(3)), c_dag('f', 0)*c('f', 1)]
                        kap = [mu[0], c_dag('f', 1)*c('f', 0)]
                        # Endpoint shape is independent of the single-particle target.
                        corr = Gf(mesh=actual.mesh_tau, target_shape=[2, 2])
                        actual._Solver__inplace_eval_one_time_correlator_topology_loop(corr, actual.G0, t, mu, kap, 1)
                        total = expected.d.get_num_single_ptcle_gf_backbones(t)
                        reference = expected.d.compute_one_time_correlator_range(expected.G0, mu, kap, expected.atom_diag, t, 0, total)
                        np.testing.assert_allclose(corr.data, reference, rtol=1e-11, atol=1e-13)
                    with self.assertRaises(TypeError): actual.d.compute_self_energy(actual.G0, t, np.array([0], np.int64))

    @unittest.skipUnless(CUDA_ENABLED, 'requires CUDA TRIQS')
    def test_collective_count_and_compute_failures(self):
        S = make_solver('cuda'); S.init_diagram_evaluator(); original = S.d
        class BadCount:
            def get_num_self_energy_backbones(self, t):
                if mpi.rank == 0: raise ValueError('injected index failure')
                return original.get_num_self_energy_backbones(t)
        S.d = BadCount()
        t = np.array([[0, 1]], np.int32)
        with self.assertRaisesRegex(RuntimeError, 'GPU index count failed'):
            S._Solver__eval_pseudo_particle_self_energy_topology_loop(S.G0, t)
        mpi.barrier(); S.d = original
        if mpi.rank == 0: S.block_sparse_workspace_budget = 65536
        S.init_diagram_evaluator()
        with self.assertRaisesRegex(RuntimeError, 'GPU evaluation failed'):
            S._Solver__eval_pseudo_particle_self_energy_topology_loop(S.G0, t)
        mpi.barrier(); S.block_sparse_workspace_budget = 1024*1024*1024; S.init_diagram_evaluator()
        value = S._Solver__eval_pseudo_particle_self_energy_topology_loop(S.G0, t)
        self.assertTrue(all(np.isfinite(value[b].data).all() for b in range(len(value))))

    @unittest.skipUnless(CUDA_ENABLED, 'requires CUDA TRIQS')
    def test_self_consistency_and_initialization_failure(self):
        from triqs_xca import _triqs_xca_cuda as runtime
        actual, expected = make_solver('cuda'), make_solver()
        if mpi.rank == 0: actual.gpu_device = runtime.cuda_device_count()
        with self.assertRaisesRegex(RuntimeError, 'Block-sparse GPU initialization failed'):
            actual.solve(max_order=1, maxiter=1, hyb_comp=False, verbose=False)
        mpi.barrier(); actual.gpu_device = None
        for S in (actual, expected):
            S.solve(max_order=1, maxiter=2, hyb_comp=False, verbose=False)
        for b in range(len(actual.G)):
            np.testing.assert_allclose(actual.G[b].data, expected.G[b].data, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(actual.Sigma[b].data, expected.Sigma[b].data, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(actual.G_tau['f'].data, expected.G_tau['f'].data, rtol=1e-9, atol=1e-12)


if __name__ == '__main__': unittest.main()
