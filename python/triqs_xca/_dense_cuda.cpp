#include <c2py/c2py.hpp>
#include "cuda_runtime.hpp"
#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>

// Host-facing binding facade; all arithmetic remains in the retained evaluator.
// Its process-wide NDA solver serialization requirement remains unchanged.
namespace xca_python {
class LibraryCudaEvaluator {
  std::unique_ptr<triqs_xca::dense::cuda::DiagramEvaluator> evaluator_;
public:
  using gf_t = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  using gf_vt = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
  template<bool C> LibraryCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex,3> coefficients,
    triqs::mesh::dlr_imtime mesh, triqs::atom_diag::atom_diag<C> const &ad)
    : evaluator_(std::make_unique<triqs_xca::dense::cuda::DiagramEvaluator>(poles,coefficients,mesh,ad)) {}
  template<bool C> LibraryCudaEvaluator(nda::vector_const_view<double> poles, nda::array_const_view<nda::dcomplex,3> coefficients,
    triqs::mesh::dlr_imtime mesh, triqs::atom_diag::atom_diag<C> const &ad,
    std::vector<triqs::operators::many_body_operator_complex> const &operators, nda::array_const_view<nda::dcomplex,3> interactions)
    : evaluator_(std::make_unique<triqs_xca::dense::cuda::DiagramEvaluator>(poles,coefficients,mesh,ad,operators,interactions)) {}
  long get_num_self_energy_backbones(nda::array_const_view<int,2> topology) { return evaluator_->get_num_self_energy_backbones(topology); }
  long get_num_single_ptcle_gf_backbones(nda::array_const_view<int,2> topology) { return evaluator_->get_num_single_ptcle_gf_backbones(topology); }
#define GF_METHOD(NAME) \
  gf_t NAME(gf_vt g,nda::array_const_view<int,2> t) { return evaluator_->NAME(g,t); } \
  gf_t NAME(gf_vt g,nda::array_const_view<int,2> t,long index) { return evaluator_->NAME(g,t,index); } \
  gf_t NAME(gf_vt g,nda::array_const_view<int,2> t,nda::array_const_view<long,1> indices) { return evaluator_->NAME(g,t,indices); }
  GF_METHOD(compute_self_energy)
  GF_METHOD(compute_self_energy_by_pairs)
#undef GF_METHOD
#define ARRAY_METHOD(NAME) \
  nda::array<nda::dcomplex,3> NAME(gf_vt g,nda::array_const_view<int,2> t) { return evaluator_->NAME(g,t); } \
  nda::array<nda::dcomplex,3> NAME(gf_vt g,nda::array_const_view<int,2> t,long index) { return evaluator_->NAME(g,t,index); } \
  nda::array<nda::dcomplex,3> NAME(gf_vt g,nda::array_const_view<int,2> t,nda::array_const_view<long,1> indices) { return evaluator_->NAME(g,t,indices); }
  ARRAY_METHOD(compute_single_ptcle_gf)
#undef ARRAY_METHOD
  template<bool C> nda::array<nda::dcomplex,3> compute_one_time_correlator(gf_vt g,
    std::vector<triqs::operators::many_body_operator_complex> const &tau,
    std::vector<triqs::operators::many_body_operator_complex> const &zero,
    triqs::atom_diag::atom_diag<C> const &ad,nda::array_const_view<int,2> t,nda::array_const_view<long,1> indices) {
    return evaluator_->compute_one_time_correlator(g,tau,zero,ad,t,indices);
  }
  std::size_t workspace_bytes() const { return evaluator_->workspace_bytes(); }
};
}
namespace c2py_module { using DenseCudaDiagramEvaluator = xca_python::LibraryCudaEvaluator; }
using evaluator = xca_python::LibraryCudaEvaluator;
using gf_view = evaluator::gf_vt;
using operator_list = std::vector<triqs::operators::many_body_operator_complex>;
#define BIND(C, PREFIX) \
PREFIX template evaluator::LibraryCudaEvaluator(nda::vector_const_view<double>,nda::array_const_view<nda::dcomplex,3>,triqs::mesh::dlr_imtime,triqs::atom_diag::atom_diag<C> const &); \
PREFIX template evaluator::LibraryCudaEvaluator(nda::vector_const_view<double>,nda::array_const_view<nda::dcomplex,3>,triqs::mesh::dlr_imtime,triqs::atom_diag::atom_diag<C> const &,operator_list const &,nda::array_const_view<nda::dcomplex,3>); \
PREFIX template nda::array<nda::dcomplex,3> evaluator::compute_one_time_correlator(gf_view,operator_list const &,operator_list const &,triqs::atom_diag::atom_diag<C> const &,nda::array_const_view<int,2>,nda::array_const_view<long,1>);
BIND(true, extern)
BIND(false, extern)
#ifndef CLAIR_C2PY_WRAP_GEN
BIND(true, )
BIND(false, )
#endif
#undef BIND
#include "_dense_cuda.wrap.cxx"
