
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0                                            = xca_python::LibraryCudaEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca._dense_cuda.DenseCudaDiagramEvaluator";
static auto _c2py_init_0                                     = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &>("poles", "coefficients", "mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("poles", "coefficients", "mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators", "interactions"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators", "interactions")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_0>    = c2py::pyfkw_constructor<_c2py_init_0>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(R"DOC()DOC");
// compute_one_time_correlator
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(g, tau, zero, ad, t, indices); },
      "self", "g", "tau", "zero", "ad", "t", "indices"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(g, tau, zero, ad, t, indices); },
      "self", "g", "tau", "zero", "ad", "t", "indices")};

// compute_self_energy
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
         -> decltype(auto) { return self.compute_self_energy(g, t); },
      "self", "g", "t"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long index) -> decltype(auto) { return self.compute_self_energy(g, t, index); },
      "self", "g", "t", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy(g, t, indices); },
      "self", "g", "t", "indices")};

// compute_self_energy_by_pairs
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, t); },
      "self", "g", "t"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long index) -> decltype(auto) { return self.compute_self_energy_by_pairs(g, t, index); },
      "self", "g", "t", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, t, indices); },
      "self", "g", "t", "indices")};

// compute_single_ptcle_gf
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, t); },
      "self", "g", "t"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long index) -> decltype(auto) { return self.compute_single_ptcle_gf(g, t, index); },
      "self", "g", "t", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, t, indices); },
      "self", "g", "t", "indices")};

// get_num_self_energy_backbones
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// workspace_bytes
static auto const _c2py_fun_6 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.workspace_bytes(); }, "self")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"compute_self_energy_by_pairs", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

// cuda_current_device
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::cuda_current_device(); })};

// cuda_device_count
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::cuda_device_count(); })};

// cuda_select_device
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cfun([](int device) { return xca_python::cuda_select_device(device); }, "device")};

static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8 = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9 = _c2py_fun_9.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"cuda_current_device", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {"cuda_device_count", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {"cuda_select_device", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT,
   "_dense_cuda",                                                                              /* name of module */
   R"RAWDOC(Retained library CUDA evaluator; serialize NDA solver use in the process.)RAWDOC", /* module documentation, may be NULL */
   -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
   module_methods,
   NULL,
   NULL,
   NULL,
   NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit__dense_cuda() {

  if (not c2py::check_python_version("_dense_cuda")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "DenseCudaDiagramEvaluator");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
