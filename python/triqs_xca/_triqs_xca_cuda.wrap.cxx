
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
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca._triqs_xca_cuda.DenseCudaDiagramEvaluator";
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

// compute_one_time_correlator_range
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long first, long last) -> decltype(auto) { return self.template compute_one_time_correlator_range<true>(g, tau, zero, ad, t, first, last); },
      "self", "g", "tau", "zero", "ad", "t", "first", "last"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long first,
         long last) -> decltype(auto) { return self.template compute_one_time_correlator_range<false>(g, tau, zero, ad, t, first, last); },
      "self", "g", "tau", "zero", "ad", "t", "first", "last")};

// compute_self_energy
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
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
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{
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

// compute_self_energy_by_pairs_range
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_by_pairs_range(g, t, first, last); },
   "self", "g", "t", "first", "last")};

// compute_self_energy_range
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_range(g, t, first, last); },
   "self", "g", "t", "first", "last")};

// compute_single_ptcle_gf
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{
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

// compute_single_ptcle_gf_range
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, xca_python::LibraryCudaEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
      long first, long last) -> decltype(auto) { return self.compute_single_ptcle_gf_range(g, t, first, last); },
   "self", "g", "t", "first", "last")};

// get_num_self_energy_backbones
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// workspace_bytes
static auto const _c2py_fun_10 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.workspace_bytes(); }, "self")};

static const auto _c2py_doc_0  = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1  = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2  = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3  = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4  = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5  = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6  = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7  = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8  = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9  = _c2py_fun_9.doc(R"DOC()DOC");
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"compute_one_time_correlator_range", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"compute_self_energy_by_pairs", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"compute_self_energy_by_pairs_range", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"compute_self_energy_range", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {"compute_single_ptcle_gf_range", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
   {"workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_10>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;
// --------- class _c2py_cls_1 -----------
using _c2py_cls_1                                            = xca_python::BlockSparseCudaEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_1>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_1> = "triqs_xca._triqs_xca_cuda.BlockSparseCudaDiagramEvaluator";
static auto _c2py_init_1                                     = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                          _c2py_cls_1,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, unsigned long, unsigned long, std::string, std::string, std::string,
                                          unsigned long, std::string>("poles", "coefficients", "mesh", "ad", "device_budget"_a = 1024ULL * 1024 * 1024,
                                                                      "sparse_budget"_a = 32ULL * 1024 * 1024, "operator_backend"_a = "library", "special_backend"_a = "reference",
                                                                      "output_placement"_a = "resident", "host_output_budget"_a = 512ULL * 1024 * 1024, "nca_backend"_a = "reference"),
   c2py::c_constructor<
                                          _c2py_cls_1,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, unsigned long, unsigned long, std::string, std::string, std::string,
                                          unsigned long, std::string>("poles", "coefficients", "mesh", "ad", "device_budget"_a = 1024ULL * 1024 * 1024,
                                                                      "sparse_budget"_a = 32ULL * 1024 * 1024, "operator_backend"_a = "library", "special_backend"_a = "reference",
                                                                      "output_placement"_a = "resident", "host_output_budget"_a = 512ULL * 1024 * 1024, "nca_backend"_a = "reference"),
   c2py::c_constructor<
                                          _c2py_cls_1,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          unsigned long, unsigned long, std::string, std::string, std::string, unsigned long, std::string>(
      "poles", "coefficients", "mesh", "ad", "operators", "interactions", "device_budget"_a = 1024ULL * 1024 * 1024,
      "sparse_budget"_a = 32ULL * 1024 * 1024, "operator_backend"_a = "library", "special_backend"_a = "reference", "output_placement"_a = "resident",
      "host_output_budget"_a = 512ULL * 1024 * 1024, "nca_backend"_a = "reference"),
   c2py::c_constructor<
                                          _c2py_cls_1,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          unsigned long, unsigned long, std::string, std::string, std::string, unsigned long, std::string>(
      "poles", "coefficients", "mesh", "ad", "operators", "interactions", "device_budget"_a = 1024ULL * 1024 * 1024,
      "sparse_budget"_a = 32ULL * 1024 * 1024, "operator_backend"_a = "library", "special_backend"_a = "reference", "output_placement"_a = "resident",
      "host_output_budget"_a = 512ULL * 1024 * 1024, "nca_backend"_a = "reference")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_1>    = c2py::pyfkw_constructor<_c2py_init_1>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_1> = _c2py_init_1.doc(R"DOC()DOC");
// compute_one_time_correlator_range
static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long first, long last) -> decltype(auto) { return self.template compute_one_time_correlator_range<true>(g, tau, zero, ad, t, first, last); },
      "self", "g", "tau", "zero", "ad", "t", "first", "last"),
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g, const std::vector<triqs::operators::many_body_operator_complex> &tau,
         const std::vector<triqs::operators::many_body_operator_complex> &zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long first,
         long last) -> decltype(auto) { return self.template compute_one_time_correlator_range<false>(g, tau, zero, ad, t, first, last); },
      "self", "g", "tau", "zero", "ad", "t", "first", "last")};

// compute_self_energy
static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
         -> decltype(auto) { return self.compute_self_energy(g, t); },
      "self", "g", "t"),
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long index) -> decltype(auto) { return self.compute_self_energy(g, t, index); },
      "self", "g", "t", "index")};

// compute_self_energy_range
static auto const _c2py_fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_range(g, t, first, last); },
   "self", "g", "t", "first", "last")};

// compute_single_ptcle_gf
static auto const _c2py_fun_14 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, t); },
      "self", "g", "t"),
   c2py::cmethod(
      [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
         long index) -> decltype(auto) { return self.compute_single_ptcle_gf(g, t, index); },
      "self", "g", "t", "index")};

// compute_single_ptcle_gf_range
static auto const _c2py_fun_15 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_1 &self, xca_python::BlockSparseCudaEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t,
      long first, long last) -> decltype(auto) { return self.compute_single_ptcle_gf_range(g, t, first, last); },
   "self", "g", "t", "first", "last")};

// correlator_storage_bound
static auto const _c2py_fun_16 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_1 const &self, long mu, long kap) -> decltype(auto) { return self.correlator_storage_bound(mu, kap); }, "self", "mu", "kap")};

// get_num_self_energy_backbones
static auto const _c2py_fun_17 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_1 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
      -> decltype(auto) { return self.get_num_self_energy_backbones(t); },
   "self", "t")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_18 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_1 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> t)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(t); },
   "self", "t")};

// last_nca_batch_paths
static auto const _c2py_fun_19 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_1 const &self) -> decltype(auto) { return self.last_nca_batch_paths(); }, "self")};

// self_energy_storage_bound
static auto const _c2py_fun_20 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_1 const &self) -> decltype(auto) { return self.self_energy_storage_bound(); }, "self"),
   c2py::cmethod(
      [](_c2py_cls_1 const &self,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.self_energy_storage_bound(topology); },
      "self", "topology")};

static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");
static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
static const auto _c2py_doc_15 = _c2py_fun_15.doc(R"DOC()DOC");
static const auto _c2py_doc_16 = _c2py_fun_16.doc(R"DOC()DOC");
static const auto _c2py_doc_17 = _c2py_fun_17.doc(R"DOC()DOC");
static const auto _c2py_doc_18 = _c2py_fun_18.doc(R"DOC()DOC");
static const auto _c2py_doc_19 = _c2py_fun_19.doc(R"DOC()DOC");
static const auto _c2py_doc_20 = _c2py_fun_20.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_1>[] = {
   {"compute_one_time_correlator_range", (PyCFunction)c2py::pyfkw<_c2py_fun_11>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_12>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_12.c_str()},
   {"compute_self_energy_range", (PyCFunction)c2py::pyfkw<_c2py_fun_13>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_14>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
   {"compute_single_ptcle_gf_range", (PyCFunction)c2py::pyfkw<_c2py_fun_15>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_15.c_str()},
   {"correlator_storage_bound", (PyCFunction)c2py::pyfkw<_c2py_fun_16>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_16.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_17>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_17.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_18>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_18.c_str()},
   {"last_nca_batch_paths", (PyCFunction)c2py::pyfkw<_c2py_fun_19>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_19.c_str()},
   {"self_energy_storage_bound", (PyCFunction)c2py::pyfkw<_c2py_fun_20>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_20.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_1> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_1>;
#ifdef XCA_HAVE_CUDA_SMALL
// --------- class _c2py_cls_2 -----------
using _c2py_cls_2                                            = triqs_xca::dense::cuda_small::DiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_2>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_2> = "triqs_xca._triqs_xca_cuda.DenseCudaSmallDiagramEvaluator";
static auto _c2py_init_2                                     = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                          _c2py_cls_2,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &>("poles", "coefficients", "mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_2,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("poles", "coefficients", "mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_2,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators",
                                                                                                              "interaction_coefficients"),
   c2py::c_constructor<
                                          _c2py_cls_2,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators",
                                                                                                              "interaction_coefficients")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_2>    = c2py::pyfkw_constructor<_c2py_init_2>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_2> = _c2py_init_2.doc(R"DOC()DOC");
// batch_capacity
static auto const _c2py_fun_21 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.batch_capacity(); }, "self")};

// batch_chunks
static auto const _c2py_fun_22 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.batch_chunks(); }, "self")};

// compute_one_time_correlator
static auto const _c2py_fun_23 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(g, ops_tau, ops_zero, ad, topology, indices); },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "indices"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(g, ops_tau, ops_zero, ad, topology, indices); },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "indices")};

// compute_one_time_correlator_range
static auto const _c2py_fun_24 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long first, long last) -> decltype(auto) {
        return self.template compute_one_time_correlator_range<true>(g, ops_tau, ops_zero, ad, topology, first, last);
      },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "first", "last"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long first, long last) -> decltype(auto) {
        return self.template compute_one_time_correlator_range<false>(g, ops_tau, ops_zero, ad, topology, first, last);
      },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "first", "last")};

// compute_self_energy
static auto const _c2py_fun_25 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_self_energy(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_self_energy_by_pairs
static auto const _c2py_fun_26 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_self_energy_by_pairs_range
static auto const _c2py_fun_27 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_by_pairs_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// compute_self_energy_range
static auto const _c2py_fun_28 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// compute_single_ptcle_gf
static auto const _c2py_fun_29 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_single_ptcle_gf_range
static auto const _c2py_fun_30 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_2 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_single_ptcle_gf_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// device_memory_reserve
static auto const _c2py_fun_31 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.device_memory_reserve(); }, "self")};

// device_workspace_bytes
static auto const _c2py_fun_32 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.device_workspace_bytes(); }, "self")};

// dimension
static auto const _c2py_fun_33 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.dimension(); }, "self")};

// get_num_self_energy_backbones
static auto const _c2py_fun_34 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_2 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_35 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_2 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// host_workspace_bound_bytes
static auto const _c2py_fun_36 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.host_workspace_bound_bytes(); }, "self")};

// num_interactions
static auto const _c2py_fun_37 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.num_interactions(); }, "self")};

// num_orbitals
static auto const _c2py_fun_38 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.num_orbitals(); }, "self")};

// rank
static auto const _c2py_fun_39 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.rank(); }, "self")};

// set_batch_capacity
static auto const _c2py_fun_40 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_2 &self, int capacity) -> decltype(auto) { return self.set_batch_capacity(capacity); }, "self", "capacity")};

// set_device_memory_reserve
static auto const _c2py_fun_41 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_2 &self, unsigned long bytes) -> decltype(auto) { return self.set_device_memory_reserve(bytes); }, "self", "bytes")};

// set_workspace_budget
static auto const _c2py_fun_42 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_2 &self, unsigned long bytes) -> decltype(auto) { return self.set_workspace_budget(bytes); }, "self", "bytes")};

// unified_workspace_bytes
static auto const _c2py_fun_43 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.unified_workspace_bytes(); }, "self")};

// workspace_budget
static auto const _c2py_fun_44 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.workspace_budget(); }, "self")};

// workspace_bytes
static auto const _c2py_fun_45 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_2 const &self) -> decltype(auto) { return self.workspace_bytes(); }, "self")};

static const auto _c2py_doc_21 = _c2py_fun_21.doc(R"DOC()DOC");
static const auto _c2py_doc_22 = _c2py_fun_22.doc(R"DOC()DOC");
static const auto _c2py_doc_23 = _c2py_fun_23.doc(R"DOC()DOC");
static const auto _c2py_doc_24 = _c2py_fun_24.doc(R"DOC()DOC");
static const auto _c2py_doc_25 = _c2py_fun_25.doc(R"DOC()DOC");
static const auto _c2py_doc_26 = _c2py_fun_26.doc(R"DOC()DOC");
static const auto _c2py_doc_27 = _c2py_fun_27.doc(R"DOC()DOC");
static const auto _c2py_doc_28 = _c2py_fun_28.doc(R"DOC()DOC");
static const auto _c2py_doc_29 = _c2py_fun_29.doc(R"DOC()DOC");
static const auto _c2py_doc_30 = _c2py_fun_30.doc(R"DOC()DOC");
static const auto _c2py_doc_31 = _c2py_fun_31.doc(R"DOC()DOC");
static const auto _c2py_doc_32 = _c2py_fun_32.doc(R"DOC()DOC");
static const auto _c2py_doc_33 = _c2py_fun_33.doc(R"DOC()DOC");
static const auto _c2py_doc_34 = _c2py_fun_34.doc(R"DOC()DOC");
static const auto _c2py_doc_35 = _c2py_fun_35.doc(R"DOC()DOC");
static const auto _c2py_doc_36 = _c2py_fun_36.doc(R"DOC()DOC");
static const auto _c2py_doc_37 = _c2py_fun_37.doc(R"DOC()DOC");
static const auto _c2py_doc_38 = _c2py_fun_38.doc(R"DOC()DOC");
static const auto _c2py_doc_39 = _c2py_fun_39.doc(R"DOC()DOC");
static const auto _c2py_doc_40 = _c2py_fun_40.doc(R"DOC()DOC");
static const auto _c2py_doc_41 = _c2py_fun_41.doc(R"DOC()DOC");
static const auto _c2py_doc_42 = _c2py_fun_42.doc(R"DOC()DOC");
static const auto _c2py_doc_43 = _c2py_fun_43.doc(R"DOC()DOC");
static const auto _c2py_doc_44 = _c2py_fun_44.doc(R"DOC()DOC");
static const auto _c2py_doc_45 = _c2py_fun_45.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_2>[] = {
   {"batch_capacity", (PyCFunction)c2py::pyfkw<_c2py_fun_21>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_21.c_str()},
   {"batch_chunks", (PyCFunction)c2py::pyfkw<_c2py_fun_22>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_22.c_str()},
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_23>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_23.c_str()},
   {"compute_one_time_correlator_range", (PyCFunction)c2py::pyfkw<_c2py_fun_24>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_24.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_25>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_25.c_str()},
   {"compute_self_energy_by_pairs", (PyCFunction)c2py::pyfkw<_c2py_fun_26>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_26.c_str()},
   {"compute_self_energy_by_pairs_range", (PyCFunction)c2py::pyfkw<_c2py_fun_27>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_27.c_str()},
   {"compute_self_energy_range", (PyCFunction)c2py::pyfkw<_c2py_fun_28>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_28.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_29>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_29.c_str()},
   {"compute_single_ptcle_gf_range", (PyCFunction)c2py::pyfkw<_c2py_fun_30>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_30.c_str()},
   {"device_memory_reserve", (PyCFunction)c2py::pyfkw<_c2py_fun_31>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_31.c_str()},
   {"device_workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_32>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_32.c_str()},
   {"dimension", (PyCFunction)c2py::pyfkw<_c2py_fun_33>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_33.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_34>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_34.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_35>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_35.c_str()},
   {"host_workspace_bound_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_36>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_36.c_str()},
   {"num_interactions", (PyCFunction)c2py::pyfkw<_c2py_fun_37>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_37.c_str()},
   {"num_orbitals", (PyCFunction)c2py::pyfkw<_c2py_fun_38>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_38.c_str()},
   {"rank", (PyCFunction)c2py::pyfkw<_c2py_fun_39>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_39.c_str()},
   {"set_batch_capacity", (PyCFunction)c2py::pyfkw<_c2py_fun_40>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_40.c_str()},
   {"set_device_memory_reserve", (PyCFunction)c2py::pyfkw<_c2py_fun_41>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_41.c_str()},
   {"set_workspace_budget", (PyCFunction)c2py::pyfkw<_c2py_fun_42>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_42.c_str()},
   {"unified_workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_43>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_43.c_str()},
   {"workspace_budget", (PyCFunction)c2py::pyfkw<_c2py_fun_44>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_44.c_str()},
   {"workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_45>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_45.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_2> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_2>;

#endif

// ==================== module functions ====================

// block_sparse_mapped_available
static auto const _c2py_fun_46 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::block_sparse_mapped_available(); })};

// block_sparse_ordered_available
static auto const _c2py_fun_47 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::block_sparse_ordered_available(); })};

// block_sparse_real_nca_available
static auto const _c2py_fun_48 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::block_sparse_real_nca_available(); })};

// cuda_current_device
static auto const _c2py_fun_49 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::cuda_current_device(); })};

// cuda_device_count
static auto const _c2py_fun_50 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return xca_python::cuda_device_count(); })};

// cuda_select_device
static auto const _c2py_fun_51 = c2py::dispatcher_f_kw_t{c2py::cfun([](int device) { return xca_python::cuda_select_device(device); }, "device")};

static const auto _c2py_doc_46 = _c2py_fun_46.doc(R"DOC()DOC");
static const auto _c2py_doc_47 = _c2py_fun_47.doc(R"DOC()DOC");
static const auto _c2py_doc_48 = _c2py_fun_48.doc(R"DOC()DOC");
static const auto _c2py_doc_49 = _c2py_fun_49.doc(R"DOC()DOC");
static const auto _c2py_doc_50 = _c2py_fun_50.doc(R"DOC()DOC");
static const auto _c2py_doc_51 = _c2py_fun_51.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"block_sparse_mapped_available", (PyCFunction)c2py::pyfkw<_c2py_fun_46>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_46.c_str()},
   {"block_sparse_ordered_available", (PyCFunction)c2py::pyfkw<_c2py_fun_47>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_47.c_str()},
   {"block_sparse_real_nca_available", (PyCFunction)c2py::pyfkw<_c2py_fun_48>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_48.c_str()},
   {"cuda_current_device", (PyCFunction)c2py::pyfkw<_c2py_fun_49>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_49.c_str()},
   {"cuda_device_count", (PyCFunction)c2py::pyfkw<_c2py_fun_50>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_50.c_str()},
   {"cuda_select_device", (PyCFunction)c2py::pyfkw<_c2py_fun_51>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_51.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "_triqs_xca_cuda",                                                   /* name of module */
                                        R"RAWDOC(CUDA diagram evaluators and shared device runtime.)RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit__triqs_xca_cuda() {

  if (not c2py::check_python_version("_triqs_xca_cuda")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_1>) < 0) return NULL;
#ifdef XCA_HAVE_CUDA_SMALL
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_2>) < 0) return NULL;
#endif

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "DenseCudaDiagramEvaluator");
  _add_type(_c2py_cls_1, "BlockSparseCudaDiagramEvaluator");
#ifdef XCA_HAVE_CUDA_SMALL
  _add_type(_c2py_cls_2, "DenseCudaSmallDiagramEvaluator");
#endif
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
