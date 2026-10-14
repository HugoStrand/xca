#include <c2py/c2py.hpp>
#include "cuda_runtime.hpp"
#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
// Generate the complete binding schema even for library-only CUDA builds.
// guard_cuda_wrappers.py guards the optional class in the generated output.
#if defined(CLAIR_C2PY_WRAP_GEN) && !defined(XCA_HAVE_CUDA_SMALL)
#define XCA_HAVE_CUDA_SMALL
#endif
#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>

// Host-facing binding facade; all arithmetic remains in the retained evaluator.
// Its process-wide NDA solver serialization requirement remains unchanged.
namespace xca_python {
  class LibraryCudaEvaluator {
    std::unique_ptr<triqs_xca::dense::cuda::DiagramEvaluator> evaluator_;

    public:
    using gf_t  = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
    template <bool C>
    LibraryCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                         triqs::atom_diag::atom_diag<C> const &ad)
       : evaluator_(std::make_unique<triqs_xca::dense::cuda::DiagramEvaluator>(poles, coefficients, mesh, ad)) {}
    template <bool C>
    LibraryCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                         triqs::atom_diag::atom_diag<C> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &operators,
                         nda::array_const_view<nda::dcomplex, 3> interactions)
       : evaluator_(std::make_unique<triqs_xca::dense::cuda::DiagramEvaluator>(poles, coefficients, mesh, ad, operators, interactions)) {}
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) { return evaluator_->get_num_self_energy_backbones(topology); }
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) { return evaluator_->get_num_single_ptcle_gf_backbones(topology); }
#define GF_METHOD(NAME)                                                                                                                              \
  gf_t NAME(gf_vt g, nda::array_const_view<int, 2> t) { return evaluator_->NAME(g, t); }                                                             \
  gf_t NAME(gf_vt g, nda::array_const_view<int, 2> t, long index) { return evaluator_->NAME(g, t, index); }                                          \
  gf_t NAME(gf_vt g, nda::array_const_view<int, 2> t, nda::array_const_view<long, 1> indices) { return evaluator_->NAME(g, t, indices); }            \
  gf_t NAME##_range(gf_vt g, nda::array_const_view<int, 2> t, long first, long last) { return evaluator_->NAME##_range(g, t, first, last); }
    GF_METHOD(compute_self_energy)
    GF_METHOD(compute_self_energy_by_pairs)
#undef GF_METHOD
#define ARRAY_METHOD(NAME)                                                                                                                           \
  nda::array<nda::dcomplex, 3> NAME(gf_vt g, nda::array_const_view<int, 2> t) { return evaluator_->NAME(g, t); }                                     \
  nda::array<nda::dcomplex, 3> NAME(gf_vt g, nda::array_const_view<int, 2> t, long index) { return evaluator_->NAME(g, t, index); }                  \
  nda::array<nda::dcomplex, 3> NAME(gf_vt g, nda::array_const_view<int, 2> t, nda::array_const_view<long, 1> indices) {                              \
    return evaluator_->NAME(g, t, indices);                                                                                                          \
  }                                                                                                                                                  \
  nda::array<nda::dcomplex, 3> NAME##_range(gf_vt g, nda::array_const_view<int, 2> t, long first, long last) {                                       \
    return evaluator_->NAME##_range(g, t, first, last);                                                                                              \
  }
    ARRAY_METHOD(compute_single_ptcle_gf)
#undef ARRAY_METHOD
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &tau,
                                                             std::vector<triqs::operators::many_body_operator_complex> const &zero,
                                                             triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> t,
                                                             nda::array_const_view<long, 1> indices) {
      return evaluator_->compute_one_time_correlator(g, tau, zero, ad, t, indices);
    }
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator_range(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &tau,
                                                                   std::vector<triqs::operators::many_body_operator_complex> const &zero,
                                                                   triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> t,
                                                                   long first, long last) {
      return evaluator_->compute_one_time_correlator_range(g, tau, zero, ad, t, first, last);
    }
    std::size_t workspace_bytes() const { return evaluator_->workspace_bytes(); }
  };
} // namespace xca_python
namespace c2py_module {
  using DenseCudaDiagramEvaluator = xca_python::LibraryCudaEvaluator;
}
namespace xca_python {
  class BlockSparseCudaEvaluator {
    using native = triqs_xca::block_sparse::cuda::DiagramEvaluator;
    std::unique_ptr<native> evaluator_;
    static triqs_xca::block_sparse::cuda::SparseBackend select_backend(std::string const &name) {
      using namespace triqs_xca::block_sparse::cuda;
      if (name == "library") return SparseBackend::Library;
      if (name != "ordered") throw std::invalid_argument("operator_backend must be library or ordered");
      if (!ordered_sparse_available()) throw std::runtime_error("Ordered block-sparse CUDA kernels are unavailable in this build");
      return SparseBackend::Ordered;
    }

    static triqs_xca::block_sparse::cuda::SpecialBackend select_special(std::string const &name) {
      using namespace triqs_xca::block_sparse::cuda;
      if (name == "reference") return SpecialBackend::Reference;
      if (name != "mapped") throw std::invalid_argument("special_backend must be reference or mapped");
      if (!mapped_special_available()) throw std::runtime_error("Mapped special-vertex CUDA kernels are unavailable in this build");
      return SpecialBackend::Mapped;
    }

    public:
    using gf_t  = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
    template <bool C>
    BlockSparseCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                             triqs::atom_diag::atom_diag<C> const &ad, std::size_t device_budget = 1024ULL * 1024 * 1024,
                             std::size_t sparse_budget = 32ULL * 1024 * 1024, std::string operator_backend = "library",
                             std::string special_backend = "reference") {
      auto selected = select_backend(operator_backend);
      auto special  = select_special(special_backend);
      evaluator_ = std::make_unique<native>(triqs_xca::block_sparse::cuda::export_model(ad, coefficients), poles, mesh, device_budget, sparse_budget,
                                            selected, special);
    }
    template <bool C>
    BlockSparseCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::mesh::dlr_imtime mesh,
                             triqs::atom_diag::atom_diag<C> const &ad, std::vector<triqs::operators::many_body_operator_complex> const &operators,
                             nda::array_const_view<nda::dcomplex, 3> interactions, std::size_t device_budget = 1024ULL * 1024 * 1024,
                             std::size_t sparse_budget = 32ULL * 1024 * 1024, std::string operator_backend = "library",
                             std::string special_backend = "reference") {
      auto selected = select_backend(operator_backend);
      auto special  = select_special(special_backend);
      evaluator_    = std::make_unique<native>(triqs_xca::block_sparse::cuda::export_model(ad, coefficients, operators, interactions), poles, mesh,
                                               device_budget, sparse_budget, selected, special);
    }
    long get_num_self_energy_backbones(nda::array_const_view<int, 2> t) { return evaluator_->get_num_self_energy_backbones(t); }
    long get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> t) { return evaluator_->get_num_single_ptcle_gf_backbones(t); }
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> t) { return evaluator_->compute_self_energy(g, t); }
    gf_t compute_self_energy(gf_vt g, nda::array_const_view<int, 2> t, long index) { return evaluator_->compute_self_energy(g, t, index); }
    gf_t compute_self_energy_range(gf_vt g, nda::array_const_view<int, 2> t, long first, long last) {
      return evaluator_->compute_self_energy_range(g, t, first, last);
    }
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> t) {
      return evaluator_->compute_single_ptcle_gf(g, t);
    }
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf(gf_vt g, nda::array_const_view<int, 2> t, long index) {
      return evaluator_->compute_single_ptcle_gf(g, t, index);
    }
    nda::array<nda::dcomplex, 3> compute_single_ptcle_gf_range(gf_vt g, nda::array_const_view<int, 2> t, long first, long last) {
      return evaluator_->compute_single_ptcle_gf_range(g, t, first, last);
    }
    template <bool C>
    nda::array<nda::dcomplex, 3> compute_one_time_correlator_range(gf_vt g, std::vector<triqs::operators::many_body_operator_complex> const &tau,
                                                                   std::vector<triqs::operators::many_body_operator_complex> const &zero,
                                                                   triqs::atom_diag::atom_diag<C> const &ad, nda::array_const_view<int, 2> t,
                                                                   long first, long last) {
      return evaluator_->compute_one_time_correlator_range(g, tau, zero, ad, t, first, last);
    }
    std::size_t self_energy_storage_bound() const { return evaluator_->self_energy_storage_bound(); }
    std::size_t correlator_storage_bound(long mu, long kap) const { return evaluator_->correlator_storage_bound(mu, kap); }
  };
} // namespace xca_python
namespace xca_python {
  inline bool block_sparse_mapped_available() { return triqs_xca::block_sparse::cuda::mapped_special_available(); }
  inline bool block_sparse_ordered_available() { return triqs_xca::block_sparse::cuda::ordered_sparse_available(); }
} // namespace xca_python
namespace c2py_module {
  using BlockSparseCudaDiagramEvaluator = xca_python::BlockSparseCudaEvaluator;
  using xca_python::block_sparse_mapped_available;
  using xca_python::block_sparse_ordered_available;
} // namespace c2py_module
using block_evaluator = xca_python::BlockSparseCudaEvaluator;
#define BLOCK_BIND(C, PREFIX)                                                                                                                        \
  PREFIX template block_evaluator::BlockSparseCudaEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>,                 \
                                                            triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &, std::size_t,            \
                                                            std::size_t, std::string, std::string);                                                  \
  PREFIX template block_evaluator::BlockSparseCudaEvaluator(                                                                                         \
     nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>, triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &,       \
     std::vector<triqs::operators::many_body_operator_complex> const &, nda::array_const_view<nda::dcomplex, 3>, std::size_t, std::size_t,           \
     std::string, std::string);                                                                                                                      \
  PREFIX template nda::array<nda::dcomplex, 3> block_evaluator::compute_one_time_correlator_range(                                                   \
     triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>, std::vector<triqs::operators::many_body_operator_complex> const &,                          \
     std::vector<triqs::operators::many_body_operator_complex> const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, long, \
     long);
BLOCK_BIND(true, extern)
BLOCK_BIND(false, extern)
#ifndef CLAIR_C2PY_WRAP_GEN
BLOCK_BIND(true, )
BLOCK_BIND(false, )
#endif
#undef BLOCK_BIND
using evaluator     = xca_python::LibraryCudaEvaluator;
using gf_view       = evaluator::gf_vt;
using operator_list = std::vector<triqs::operators::many_body_operator_complex>;
#define BIND(C, PREFIX)                                                                                                                              \
  PREFIX template evaluator::LibraryCudaEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>, triqs::mesh::dlr_imtime,  \
                                                  triqs::atom_diag::atom_diag<C> const &);                                                           \
  PREFIX template evaluator::LibraryCudaEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>, triqs::mesh::dlr_imtime,  \
                                                  triqs::atom_diag::atom_diag<C> const &, operator_list const &,                                     \
                                                  nda::array_const_view<nda::dcomplex, 3>);                                                          \
  PREFIX template nda::array<nda::dcomplex, 3> evaluator::compute_one_time_correlator(                                                               \
     gf_view, operator_list const &, operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>,                   \
     nda::array_const_view<long, 1>);                                                                                                                \
  PREFIX template nda::array<nda::dcomplex, 3> evaluator::compute_one_time_correlator_range(                                                         \
     gf_view, operator_list const &, operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, long, long);
BIND(true, extern)
BIND(false, extern)
#ifndef CLAIR_C2PY_WRAP_GEN
BIND(true, )
BIND(false, )
#endif
#undef BIND

#ifdef XCA_HAVE_CUDA_SMALL
namespace c2py_module {
  using DenseCudaSmallDiagramEvaluator = triqs_xca::dense::cuda_small::DiagramEvaluator;
}
using small_evaluator     = triqs_xca::dense::cuda_small::DiagramEvaluator;
using small_gf_view       = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
using small_operator_list = std::vector<triqs::operators::many_body_operator_complex>;

#define BIND_INSTANTIATIONS(C)                                                                                                                       \
  extern template small_evaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>,                         \
                                                    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &);                                \
  extern template small_evaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex, 3>,                         \
                                                    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &, small_operator_list const &,    \
                                                    nda::array_const_view<nda::dcomplex, 3>);                                                        \
  extern template nda::array<nda::dcomplex, 3> small_evaluator::compute_one_time_correlator(                                                         \
     small_gf_view, small_operator_list const &, small_operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, \
     nda::array_const_view<long, 1>);                                                                                                                \
  extern template nda::array<nda::dcomplex, 3> small_evaluator::compute_one_time_correlator_range(                                                   \
     small_gf_view, small_operator_list const &, small_operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int, 2>, \
     long, long);
BIND_INSTANTIATIONS(true)
BIND_INSTANTIATIONS(false)
#undef BIND_INSTANTIATIONS

#endif

#include "_triqs_xca_cuda.wrap.cxx"
