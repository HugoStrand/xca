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

Select ``block_sparse_operator_backend='ordered'`` for the native ordered sparse
operator route. ``'library'`` remains the default. The ordered route requires
native kernels built with ``XCA_CUDA_BLOCK_SPARSE=AUTO`` (default, when a CUDA
compiler is available) or ``ON``. ``OFF`` preserves a library-only CUDA build;
this option is independent of ``XCA_CUDA_SMALL``. Query compiled support with
``triqs_xca.gpu.block_sparse_ordered_available()``. Explicit unsupported requests
raise before model export/device setup, including collectively during solver setup.
The consolidated C++ facade also accepts ``operator_backend='ordered'``.


Set ``block_sparse_special_backend='mapped'`` to fuse eligible special-vertex
right products, weights and ordered field additions. The reference policy remains
``'reference'`` by default. Mapping requires the same native build option and can
be selected independently of the sparse operator backend. Query support with
``triqs_xca.gpu.block_sparse_mapped_available()``. The consolidated facade accepts
``special_backend='mapped'``; C++ uses ``SpecialBackend::Mapped``.

Eligibility uses the actual sparse structure: at most one nonzero per kappa column.
Other operators and singleton sets retain the reference special vertex. Column
maps and weights add a conservative metadata allowance to the device budget;
no cache of padded kappa matrices is retained. These structural choices occur
before mapped device work, and errors never trigger CPU replay. Numerical field
order is preserved, though fused/library rounding need not be bitwise identical.
SPGF and correlators retain their existing device path. The opt-in setting and
its archive default are independent of any unqualified GH200 crossover policy.


``block_sparse_workspace_budget`` limits conservative explicit device storage
(default 1 GiB); ``block_sparse_operator_budget`` limits the cached operator and
its sparse-library workspace (default 32 MiB). Host and opaque-library storage are
additional. Larger sectors require capacity qualification and suitable budgets.
The evaluator exposes ``self_energy_storage_bound(topology)`` for a validated
schedule-specific estimate; the no-argument form remains an order-independent
upper bound. NCA omits inactive convolution scratch and releases completed DLR
scratch from earlier higher-order calls before checking available memory. Fixed
rank tables and full input/output reserves remain. Higher orders and correlators
retain their conservative working-set estimates. A topology-specific budget can
therefore admit NCA while rejecting OCA; such rejection precedes input upload and
path arithmetic and leaves later valid calls usable.

This initial reference covers self-energy, SPGF and one-time correlators; its
dispatch/batching optimizations are still under investigation. The performance
target is GH200-class hardware; workstation measurements are development proxies.

Each MPI process keeps one CUDA device. Select it before constructing evaluators,
using ``gpu_device`` or launcher-managed ``CUDA_VISIBLE_DEVICES``. Collective solver
operations coordinate setup, index-count and evaluation errors before reductions;
standalone evaluator calls remain local. No automatic CPU replay follows submitted
device work. Operator/backend preferences and budgets are retained in solver archives. Older
archives default to library operator execution. CPU evaluation can retain the
ordered preference without loading CUDA; it applies when CUDA is selected.

