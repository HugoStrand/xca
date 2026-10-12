
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
using _c2py_cls_0                                            = triqs_xca::dense::cuda_small::DiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca._dense_cuda_small.DenseCudaSmallDiagramEvaluator";
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
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators",
                                                                                                              "interaction_coefficients"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("poles", "coefficients", "mesh", "ad", "operators",
                                                                                                              "interaction_coefficients")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_0>    = c2py::pyfkw_constructor<_c2py_init_0>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(R"DOC()DOC");
// batch_capacity
static auto const _c2py_fun_0 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.batch_capacity(); }, "self")};

// batch_chunks
static auto const _c2py_fun_1 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.batch_chunks(); }, "self")};

// compute_one_time_correlator
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(g, ops_tau, ops_zero, ad, topology, indices); },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "indices"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(g, ops_tau, ops_zero, ad, topology, indices); },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "indices")};

// compute_one_time_correlator_range
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long first, long last) -> decltype(auto) {
        return self.template compute_one_time_correlator_range<true>(g, ops_tau, ops_zero, ad, topology, first, last);
      },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "first", "last"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_zero, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long first, long last) -> decltype(auto) {
        return self.template compute_one_time_correlator_range<false>(g, ops_tau, ops_zero, ad, topology, first, last);
      },
      "self", "g", "ops_tau", "ops_zero", "ad", "topology", "first", "last")};

// compute_self_energy
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_self_energy(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_self_energy_by_pairs
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_self_energy_by_pairs_range
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_by_pairs_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// compute_self_energy_range
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_self_energy_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// compute_single_ptcle_gf
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology); },
      "self", "g", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         long index) -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology, index); },
      "self", "g", "topology", "index"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const long, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> indices)
         -> decltype(auto) { return self.compute_single_ptcle_gf(g, topology, indices); },
      "self", "g", "topology", "indices")};

// compute_single_ptcle_gf_range
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self, triqs_xca::dense::cuda_small::DiagramEvaluator::gf_vt g,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      long first, long last) -> decltype(auto) { return self.compute_single_ptcle_gf_range(g, topology, first, last); },
   "self", "g", "topology", "first", "last")};

// dimension
static auto const _c2py_fun_10 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.dimension(); }, "self")};

// get_num_self_energy_backbones
static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// num_interactions
static auto const _c2py_fun_13 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.num_interactions(); }, "self")};

// num_orbitals
static auto const _c2py_fun_14 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.num_orbitals(); }, "self")};

// rank
static auto const _c2py_fun_15 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.rank(); }, "self")};

// set_batch_capacity
static auto const _c2py_fun_16 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_0 &self, int capacity) -> decltype(auto) { return self.set_batch_capacity(capacity); }, "self", "capacity")};

// set_workspace_budget
static auto const _c2py_fun_17 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_0 &self, unsigned long bytes) -> decltype(auto) { return self.set_workspace_budget(bytes); }, "self", "bytes")};

// workspace_budget
static auto const _c2py_fun_18 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.workspace_budget(); }, "self")};

// workspace_bytes
static auto const _c2py_fun_19 =
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
static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");
static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
static const auto _c2py_doc_15 = _c2py_fun_15.doc(R"DOC()DOC");
static const auto _c2py_doc_16 = _c2py_fun_16.doc(R"DOC()DOC");
static const auto _c2py_doc_17 = _c2py_fun_17.doc(R"DOC()DOC");
static const auto _c2py_doc_18 = _c2py_fun_18.doc(R"DOC()DOC");
static const auto _c2py_doc_19 = _c2py_fun_19.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"batch_capacity", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"batch_chunks", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"compute_one_time_correlator_range", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"compute_self_energy_by_pairs", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"compute_self_energy_by_pairs_range", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {"compute_self_energy_range", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {"compute_single_ptcle_gf_range", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
   {"dimension", (PyCFunction)c2py::pyfkw<_c2py_fun_10>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_11>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_12>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_12.c_str()},
   {"num_interactions", (PyCFunction)c2py::pyfkw<_c2py_fun_13>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
   {"num_orbitals", (PyCFunction)c2py::pyfkw<_c2py_fun_14>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
   {"rank", (PyCFunction)c2py::pyfkw<_c2py_fun_15>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_15.c_str()},
   {"set_batch_capacity", (PyCFunction)c2py::pyfkw<_c2py_fun_16>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_16.c_str()},
   {"set_workspace_budget", (PyCFunction)c2py::pyfkw<_c2py_fun_17>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_17.c_str()},
   {"workspace_budget", (PyCFunction)c2py::pyfkw<_c2py_fun_18>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_18.c_str()},
   {"workspace_bytes", (PyCFunction)c2py::pyfkw<_c2py_fun_19>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_19.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT,
   "_dense_cuda_small",                                                           /* name of module */
   R"RAWDOC(Native small-matrix CUDA evaluator with bounded work ranges.)RAWDOC", /* module documentation, may be NULL */
   -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
   module_methods,
   NULL,
   NULL,
   NULL,
   NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit__dense_cuda_small() {

  if (not c2py::check_python_version("_dense_cuda_small")) return NULL;

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
  _add_type(_c2py_cls_0, "DenseCudaSmallDiagramEvaluator");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
