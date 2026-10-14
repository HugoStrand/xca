.. _python_api:

Python API Reference
====================

This section contains the complete Python API reference for the triqs_xca package,
including all modules, classes, and compiled extensions.

Main Module
-----------

The main ``triqs_xca`` module provides access to all submodules and the high-level Solver interface.

High-Level Solver Interface
----------------------------

.. autoclass:: triqs_xca.Solver
   :members: solve, set_dynamic_interactions, expectation_value, partition_function, pseudo_particle_chemical_potential

CUDA evaluator selection
------------------------

CPU evaluation is the default. With a CUDA-enabled TRIQS installation, select the
block-sparse device reference while retaining atomic symmetries::

    solver = Solver(H_loc, beta, w_max, eps, gf_struct,
                    conserved_operators='automatic',
                    block_sparse_backend='cuda', gpu_device=0)

Explicit conserved operators also use this backend. Dense CUDA choices remain
``dense_backend='cuda'`` or ``'cuda_small'`` with ``conserved_operators=[]``.
All GPU bindings live in ``triqs_xca._triqs_xca_cuda`` and are loaded lazily.
The block-sparse reference requires CUDA libraries but does not require native
CUDA-small kernels.

``block_sparse_workspace_budget`` limits conservative explicit device storage
(default 1 GiB); ``block_sparse_operator_budget`` limits the cached operator and
its sparse-library workspace (default 32 MiB). Host and opaque-library storage are
additional. Larger sectors require capacity qualification and suitable budgets.
This initial reference covers self-energy, SPGF and one-time correlators; its
dispatch/batching optimizations are still under investigation. The performance
target is GH200-class hardware; workstation measurements are development proxies.

Each MPI process keeps one CUDA device. Select it before constructing evaluators,
using ``gpu_device`` or launcher-managed ``CUDA_VISIBLE_DEVICES``. Collective solver
operations coordinate setup, index-count and evaluation errors before reductions;
standalone evaluator calls remain local. No automatic CPU replay follows submitted
device work. Backend preferences and budgets are retained in solver archives.

