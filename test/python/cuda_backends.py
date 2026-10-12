import copy
import tempfile
from pathlib import Path
from h5 import HDFArchive
import unittest
from unittest.mock import patch, Mock
import numpy as np
from triqs.operators import n, c, c_dag
import triqs.utility.mpi as mpi
from triqs_xca import Solver
from triqs_xca.solver import scatter_range_bounds
from triqs_xca._backend_config import CUDA_ENABLED, CUDA_SMALL_ENABLED


def make_solver(backend='cpu'):
    H = -.2*(n('0',0)+n('0',1)) + .4*n('0',0)*n('0',1)
    S = Solver(H,2.,5.,1e-12,[['0',2]],conserved_operators=[],verbose=False,dense_backend=backend)
    coefficients=np.zeros((2,2,2),complex)
    coefficients[0]=np.eye(2)*.3
    coefficients[1]=np.eye(2)*.2
    S.set_hybridization_poles_and_coefficients(np.array([-1.5,.6]),coefficients)
    return S


class BackendTests(unittest.TestCase):
    def test_range_partition_and_collective_failure(self):
        for count in [0,1,7,17,2**60]:
            bounds=[]
            for rank in range(4):
                with patch('triqs_xca.solver.mpi.rank',rank), patch('triqs_xca.solver.mpi.size',4):
                    bounds.append(scatter_range_bounds(count))
            self.assertEqual(bounds[0][0],0)
            self.assertEqual(bounds[-1][1],count)
            for left,right in zip(bounds,bounds[1:]): self.assertEqual(left[1],right[0])
            self.assertLessEqual(max(end-start for start,end in bounds)-min(end-start for start,end in bounds),1)
        S=make_solver()
        def operation():
            if mpi.rank==0: raise ValueError('injected rank failure')
            return 0
        with self.assertRaisesRegex(RuntimeError,'Native dense evaluation failed'):
            S._native_range_call(operation)

    def test_preference_copy_and_old_archive_defaults(self):
        S=make_solver('cuda_small')
        S.gpu_device=0
        S.dense_device_memory_reserve=32*1024*1024
        if CUDA_SMALL_ENABLED: S.init_diagram_evaluator()
        self.assertNotIn('d',S.__reduce_to_dict__())
        clone=copy.deepcopy(S)
        self.assertEqual(clone.dense_backend,'cuda_small')
        self.assertEqual(clone.dense_batch_capacity,128)
        self.assertEqual(clone.gpu_device,0)
        self.assertEqual(clone.dense_device_memory_reserve,32*1024*1024)
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'solver.h5')
            with HDFArchive(path,'w') as archive: archive['solver']=S
            with HDFArchive(path,'r') as archive: restored=archive['solver']
            self.assertEqual(restored.dense_backend,'cuda_small')
            self.assertEqual(restored.dense_workspace_budget,S.dense_workspace_budget)
            self.assertEqual(restored.gpu_device,0)
            self.assertEqual(restored.dense_device_memory_reserve,32*1024*1024)
            self.assertFalse(hasattr(restored,'d'))
        data=S.__reduce_to_dict__()
        for key in ['dense_backend','dense_batch_capacity','dense_workspace_budget','gpu_device','dense_device_memory_reserve']:
            data.pop(key)
        restored=Solver.__factory_from_dict__('Solver',data)
        self.assertEqual(restored.dense_backend,'cpu')
        self.assertIsNone(restored.gpu_device)
        self.assertEqual(restored.dense_device_memory_reserve,64*1024*1024)
        self.assertFalse(hasattr(restored,'d'))

    def test_default_gpu_device_archive(self):
        S=make_solver()
        self.assertIsNone(S.gpu_device)
        self.assertNotIn('gpu_device',S.__reduce_to_dict__())
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'solver.h5')
            with HDFArchive(path,'w') as archive: archive['solver']=S
            with HDFArchive(path,'r') as archive: restored=archive['solver']
            self.assertIsNone(restored.gpu_device)
            self.assertEqual(S,restored)
        self.assertIsNone(copy.deepcopy(S).gpu_device)

    def test_invalid_configuration_and_unavailable_backend(self):
        with self.assertRaises(ValueError):
            Solver(n('0',0),2.,5.,1e-12,[['0',1]],dense_backend='cuda_small',verbose=False)
        with self.assertRaises(ValueError):
            Solver(n('0',0),2.,5.,1e-12,[['0',1]],conserved_operators=[],dense_backend='typo',verbose=False)
        if not CUDA_SMALL_ENABLED:
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                make_solver('cuda_small').init_diagram_evaluator()

    def test_gpu_device_configuration(self):
        from triqs_xca.solver import _configure_gpu_device
        for ordinal in [-1, True]:
            with self.assertRaises(ValueError):
                Solver(n('0',0),2.,5.,1e-12,[['0',1]],verbose=False,gpu_device=ordinal)
        runtime=Mock()
        runtime.cuda_device_count.return_value=2
        runtime.cuda_current_device.return_value=1
        with patch.dict(_configure_gpu_device.__globals__, {'_gpu_process_device': None}), patch('importlib.import_module',return_value=runtime):
            self.assertEqual(_configure_gpu_device('cuda_small',None),1)
            runtime.cuda_select_device.assert_called_once_with(1)
            with self.assertRaisesRegex(RuntimeError,'one unchanged CUDA device'):
                _configure_gpu_device('cuda',0)
            runtime.cuda_current_device.return_value=0
            with self.assertRaisesRegex(RuntimeError,'one unchanged CUDA device'):
                _configure_gpu_device('cuda_small',1)
            with self.assertRaisesRegex(ValueError,'unavailable'):
                _configure_gpu_device('cuda_small',2)

    def test_invalid_device_reserve(self):
        for reserve in [-1,True]:
            with self.assertRaises(ValueError):
                Solver(n('0',0),2.,5.,1e-12,[['0',1]],verbose=False,dense_device_memory_reserve=reserve)

    @unittest.skipUnless(CUDA_ENABLED,'requires CUDA TRIQS')
    def test_gpu_initialization_failure_is_collective(self):
        from triqs_xca import _dense_cuda as runtime
        for backend in ['cuda']+(['cuda_small'] if CUDA_SMALL_ENABLED else []):
            S=make_solver(backend)
            if mpi.rank==0: S.gpu_device=runtime.cuda_device_count()
            with self.assertRaisesRegex(RuntimeError,'Dense GPU initialization failed'):
                S.solve(max_order=1,maxiter=1,verbose=False,hyb_comp=False)
            mpi.barrier() # Every rank must leave the failed initialization.
            S.gpu_device=None
            S.solve(max_order=1,maxiter=1,verbose=False,hyb_comp=False)
            self.assertTrue(np.isfinite(S.G_tau['0'].data).all())

    @unittest.skipUnless(CUDA_ENABLED,'requires CUDA TRIQS')
    def test_visible_gpu_device_selection(self):
        from triqs_xca import _dense_cuda as runtime
        current=runtime.cuda_current_device()
        for backend in ['cuda']+(['cuda_small'] if CUDA_SMALL_ENABLED else []):
            S=make_solver(backend)
            S.gpu_device=current
            S.init_diagram_evaluator()
            self.assertEqual(runtime.cuda_current_device(),current)
            S=make_solver(backend)
            S.gpu_device=runtime.cuda_device_count()
            with self.assertRaisesRegex(ValueError,'unavailable'):
                S.init_diagram_evaluator()

    @unittest.skipUnless(CUDA_ENABLED,'requires CUDA TRIQS')
    def test_gpu_methods_match_cpu(self):
        reference=make_solver()
        reference.init_diagram_evaluator()
        t=np.array([[0,2],[1,3]],dtype=np.int32)
        for backend in ['cuda']+(['cuda_small'] if CUDA_SMALL_ENABLED else []):
            S=make_solver(backend)
            S.init_diagram_evaluator()
            ix=np.arange(min(17,S.d.get_num_self_energy_backbones(t)),dtype=np.int64)
            for name in ['compute_self_energy','compute_self_energy_by_pairs']:
                a=getattr(S.d,name)(S.G0,t,ix)
                b=getattr(reference.d,name)(reference.G0,t,ix)
                np.testing.assert_allclose(a[0].data,b[0].data,rtol=1e-10,atol=1e-13)
            ix=np.arange(min(9,S.d.get_num_single_ptcle_gf_backbones(t)),dtype=np.int64)
            np.testing.assert_allclose(S.d.compute_single_ptcle_gf(S.G0,t,ix),reference.d.compute_single_ptcle_gf(reference.G0,t,ix),rtol=1e-10,atol=1e-13)
            tau=[c('0',0),c('0',1)]; zero=[c_dag('0',0),c_dag('0',1)]
            np.testing.assert_allclose(S.d.compute_one_time_correlator(S.G0,tau,zero,S.atom_diag,t,ix),reference.d.compute_one_time_correlator(reference.G0,tau,zero,reference.atom_diag,t,ix),rtol=1e-10,atol=1e-13)

    @unittest.skipUnless(CUDA_SMALL_ENABLED,'requires CUDA-small')
    def test_fitted_dynamic_channels_match_library_and_cpu(self):
        # Supply fitted channel data directly to isolate evaluator integration
        # from the independent interaction fitting tests.
        solvers=[]
        for backend in ['cpu','cuda','cuda_small']:
            S=make_solver(backend)
            S.has_dynamic_interactions=True
            S.dynint_ops=[n('0',0),c_dag('0',0)*c('0',1)+c_dag('0',1)*c('0',0)]
            S.dynint_coeffs=np.array([[[.02,.004j],[-.004j,.03]],[[.015,0.],[0.,.02]]],dtype=complex)
            S.init_diagram_evaluator()
            solvers.append(S)
        reference,library,actual=solvers
        t=np.array([[0,3],[1,4],[2,5]],dtype=np.int32)
        indices=np.arange(129,dtype=np.int64)
        before=actual.d.batch_chunks()
        for name in ['compute_self_energy','compute_self_energy_by_pairs']:
            expected=getattr(library.d,name)(library.G0,t,indices)[0].data
            value=getattr(actual.d,name)(actual.G0,t,indices)[0].data
            np.testing.assert_allclose(value,expected,rtol=1e-10,atol=1e-280)
            np.testing.assert_allclose(value,getattr(reference.d,name)(reference.G0,t,indices)[0].data,rtol=1e-9,atol=1e-13)
        self.assertGreater(actual.d.batch_chunks(),before)

    @unittest.skipUnless(CUDA_SMALL_ENABLED,'requires CUDA-small')
    def test_self_consistency_matches_cpu(self):
        reference=make_solver()
        actual=make_solver('cuda_small')
        for S in [reference,actual]:
            S.solve(max_order=1,maxiter=2,tol=1e-8,spgf_max_order=1,verbose=False,hyb_comp=False)
        np.testing.assert_allclose(actual.G[0].data,reference.G[0].data,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(actual.Sigma[0].data,reference.Sigma[0].data,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(actual.G_tau['0'].data,reference.G_tau['0'].data,rtol=1e-9,atol=1e-12)

    @unittest.skipUnless(CUDA_SMALL_ENABLED,'requires CUDA-small')
    def test_solver_uses_native_ranges(self):
        S=make_solver('cuda_small'); S.init_diagram_evaluator()
        self.assertGreater(S.d.device_workspace_bytes(),0)
        self.assertGreater(S.d.unified_workspace_bytes(),0)
        self.assertGreater(S.d.host_workspace_bound_bytes(),0)
        reference=make_solver(); reference.init_diagram_evaluator()
        with patch('triqs_xca.solver.scatter_range_over_ranks',side_effect=AssertionError('materialized index array')):
            actual=S.eval_pseudo_particle_self_energy_order_by_pairs(S.G0,2)
            spgf=S.eval_single_particle_greens_function(S.G0,2)
            corr=S.eval_one_time_correlator(S.G0,2,[n('0',0)],[n('0',1)])
        expected=reference.eval_pseudo_particle_self_energy_order_by_pairs(reference.G0,2)
        np.testing.assert_allclose(actual[0].data,expected[0].data,rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(spgf.data,reference.eval_single_particle_greens_function(reference.G0,2).data,rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(corr.data,reference.eval_one_time_correlator(reference.G0,2,[n('0',0)],[n('0',1)]).data,rtol=1e-10,atol=1e-13)


if __name__=='__main__': unittest.main()
