#include <c2py/c2py.hpp>
#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>
#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>

namespace c2py_module {
  using DenseCudaSmallDiagramEvaluator = triqs_xca::dense::cuda_small::DiagramEvaluator;
}
using evaluator = triqs_xca::dense::cuda_small::DiagramEvaluator;
using gf_view = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
using operator_list = std::vector<triqs::operators::many_body_operator_complex>;

#define BIND_INSTANTIATIONS(C) \
extern template evaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex,3>, triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &); \
extern template evaluator::DiagramEvaluator(nda::vector_const_view<double>, nda::array_const_view<nda::dcomplex,3>, triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<C> const &, operator_list const &, nda::array_const_view<nda::dcomplex,3>); \
extern template nda::array<nda::dcomplex,3> evaluator::compute_one_time_correlator(gf_view, operator_list const &, operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int,2>, nda::array_const_view<long,1>); \
extern template nda::array<nda::dcomplex,3> evaluator::compute_one_time_correlator_range(gf_view, operator_list const &, operator_list const &, triqs::atom_diag::atom_diag<C> const &, nda::array_const_view<int,2>, long, long);
BIND_INSTANTIATIONS(true)
BIND_INSTANTIATIONS(false)
#undef BIND_INSTANTIATIONS

#include "_dense_cuda_small.wrap.cxx"
