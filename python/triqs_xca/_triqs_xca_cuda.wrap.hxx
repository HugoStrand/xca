#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION__triqs_xca_cuda_GUARDS
#define C2PY_HXX_DECLARATION__triqs_xca_cuda_GUARDS
template <> constexpr bool c2py::is_wrapped<xca_python::LibraryCudaEvaluator>         = true;
template <> inline constexpr auto c2py::tp_name<xca_python::LibraryCudaEvaluator>     = "triqs_xca._triqs_xca_cuda.DenseCudaDiagramEvaluator";
template <> constexpr bool c2py::is_wrapped<xca_python::BlockSparseCudaEvaluator>     = true;
template <> inline constexpr auto c2py::tp_name<xca_python::BlockSparseCudaEvaluator> = "triqs_xca._triqs_xca_cuda.BlockSparseCudaDiagramEvaluator";
#ifdef XCA_HAVE_CUDA_SMALL
template <> constexpr bool c2py::is_wrapped<triqs_xca::dense::cuda_small::DiagramEvaluator> = true;
template <>
inline constexpr auto c2py::tp_name<triqs_xca::dense::cuda_small::DiagramEvaluator> = "triqs_xca._triqs_xca_cuda.DenseCudaSmallDiagramEvaluator";
#endif
#endif