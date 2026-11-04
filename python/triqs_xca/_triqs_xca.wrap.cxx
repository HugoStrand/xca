
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

#define C2PY_VERSION_MAJOR 1
#define C2PY_VERSION_MINOR 0

#include <c2py/c2py.hpp>
#include <c2py/version_check.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_698c8c52 -----------
using _c2py_cls_698c8c52                                            = triqs_xca::dense::DiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_698c8c52>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_698c8c52> = "triqs_xca._triqs_xca.DenseDiagramEvaluator";
static const auto _c2py_init_1d003032                               = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                    _c2py_cls_698c8c52,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                    _c2py_cls_698c8c52,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                    _c2py_cls_698c8c52,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops", "dynint_coeffs"),
   c2py::c_constructor<
                                    _c2py_cls_698c8c52,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops",
                                                                                                        "dynint_coeffs")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_698c8c52> = c2py::pyfkw_constructor<_c2py_init_1d003032>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_698c8c52> = _c2py_init_1d003032.doc(
   R"DOC(
Constructor for DiagramEvaluator

Parameters
----------
hyb_poles : {par_0}
   hybridization poles
hyb_coeffs : {par_1}
   hybridization function coefficients (at poles)
tau_mesh : {par_2}
   TRIQS imagnary time DLR mesh
ad : {par_3}
   TRIQS atom_diag object with Hamiltonian and field operators
dynint_ops : {par_4}
   vector of many_body_operator_complex objects representing the dynamic interactions
dynint_coeffs : {par_5}
   array of coefficients for the dynamic interactions (also using hyb_poles)
)DOC",
   {{c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<triqs::mesh::dlr_imtime>()},
    {c2py::python_typename<const triqs::atom_diag::atom_diag<1> &>(), c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()},
    {c2py::python_typename<const std::vector<triqs::operators::many_body_operator_complex> &>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()}});
// compute_one_time_correlator
static auto const _c2py_fun_06fc17c6 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_698c8c52 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology"),
   c2py::cmethod(
      [](_c2py_cls_698c8c52 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology")};

// compute_self_energy
static auto const _c2py_fun_48d3f32f = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_698c8c52 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

// compute_single_ptcle_gf
static auto const _c2py_fun_4dca0593 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_698c8c52 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

static const auto _c2py_doc_06fc17c6 = _c2py_fun_06fc17c6.doc(R"DOC(
Evaluate this MPI rank's contribution for the supplied observable operators.
)DOC");
static const auto _c2py_doc_48d3f32f = _c2py_fun_48d3f32f.doc(R"DOC(
Evaluate this MPI rank's share of a topology, pairing the special arc's two directions.
Both directions share the dense backbone convolutions. The caller reduces the accumulated result.
)DOC");
static const auto _c2py_doc_4dca0593 = _c2py_fun_4dca0593.doc(R"DOC(
Evaluate this MPI rank's contribution to the single-particle Green's function.
)DOC");

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_698c8c52>[] = {
   PMDF("compute_one_time_correlator", 06fc17c6),
   PMDF("compute_self_energy", 48d3f32f),
   PMDF("compute_single_ptcle_gf", 4dca0593),
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

template <>
const std::string c2py::tp_doc<_c2py_cls_698c8c52> = R"DOC(Class for evaluating a diagram of a given order and topology
This class is used to evaluate all the diagram decompositions of a given
order and topology. It consumes immutable backbone descriptions and
contains the Green's functions and creation/annihilation operators needed to
actually compute the backbone. It also contains temporary arrays required for
computation.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_698c8c52>;
// --------- class _c2py_cls_f8dfe6a1 -----------
using _c2py_cls_f8dfe6a1                                            = triqs_xca::dense::DiagramEvaluatorFewOrb;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_f8dfe6a1>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_f8dfe6a1> = "triqs_xca._triqs_xca.DenseDiagramEvaluatorFewOrb";
static const auto _c2py_init_9cc046f5                               = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                    _c2py_cls_f8dfe6a1,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, std::size_t>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad",
                                                                                                                  "cache_bytes"_a = 8 * 1024 * 1024),
   c2py::c_constructor<
                                    _c2py_cls_f8dfe6a1,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, std::size_t>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad",
                                                                                                                  "cache_bytes"_a = 8 * 1024 * 1024),
   c2py::c_constructor<
                                    _c2py_cls_f8dfe6a1,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    std::size_t>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops", "dynint_coeffs", "cache_bytes"_a = 8 * 1024 * 1024),
   c2py::c_constructor<
                                    _c2py_cls_f8dfe6a1,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    std::size_t>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops", "dynint_coeffs", "cache_bytes"_a = 8 * 1024 * 1024)};
template <> constexpr initproc c2py::tp_init<_c2py_cls_f8dfe6a1> = c2py::pyfkw_constructor<_c2py_init_9cc046f5>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_f8dfe6a1> = _c2py_init_9cc046f5.doc(R"DOC(
[1, 2] Construct from physical poles, hybridization coefficients and atomic operators.
cache_bytes bounds retained edge operands; zero disables retention.

------

[3, 4] Include retarded interactions using the same physical poles as the hybridization.

------
)DOC");
// compute_one_time_correlator
static auto const _c2py_fun_8430ea53 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_f8dfe6a1 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology"),
   c2py::cmethod(
      [](_c2py_cls_f8dfe6a1 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology")};

// compute_self_energy
static auto const _c2py_fun_a1754456 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_f8dfe6a1 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

// compute_single_ptcle_gf
static auto const _c2py_fun_06b7b7d6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_f8dfe6a1 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

static const auto _c2py_doc_8430ea53 = _c2py_fun_8430ea53.doc(R"DOC(
Evaluate this MPI rank's contribution for the supplied observable operators; the caller reduces the result.
)DOC");
static const auto _c2py_doc_a1754456 = _c2py_fun_a1754456.doc(R"DOC(
Evaluate this MPI rank's share of a topology, pairing the special arc's two directions.
Both directions share the dense backbone convolutions. The caller reduces the accumulated result.
)DOC");
static const auto _c2py_doc_06b7b7d6 = _c2py_fun_06b7b7d6.doc(R"DOC(
Evaluate this MPI rank's single-particle Green's function contribution; the caller reduces the result.
)DOC");

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_f8dfe6a1>[] = {
   PMDF("compute_one_time_correlator", 8430ea53),
   PMDF("compute_self_energy", a1754456),
   PMDF("compute_single_ptcle_gf", 06b7b7d6),
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

template <>
const std::string c2py::tp_doc<_c2py_cls_f8dfe6a1> = R"DOC(Dense CPU evaluation with small matrix products and reusable edge operands.
A cache entry contains coefficients of G times an edge kernel product and
their Hilbert transform. Entries are shared within one public call only:
modifying G between calls, including in place, never requires invalidation.
Prepared calls sum locally; topology calls sum this MPI rank's contribution.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_f8dfe6a1>;
// --------- class _c2py_cls_efedd95d -----------
using _c2py_cls_efedd95d                                            = triqs_xca::block_sparse::DiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_efedd95d>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_efedd95d> = "triqs_xca._triqs_xca.BlockSparseDiagramEvaluator";
static const auto _c2py_init_823be880                               = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                    _c2py_cls_efedd95d,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                    _c2py_cls_efedd95d,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                    _c2py_cls_efedd95d,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops", "dynint_coeffs"),
   c2py::c_constructor<
                                    _c2py_cls_efedd95d,
                                    nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>,
                                    triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_complex> &,
                                    nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops",
                                                                                                        "dynint_coeffs")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_efedd95d> = c2py::pyfkw_constructor<_c2py_init_823be880>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_efedd95d> = _c2py_init_823be880.doc(
   R"DOC(
[1, 2] Constructor for DiagramEvaluator

------

[3, 4] Constructor for DiagramEvaluator with dynamical interactions

The interaction operators extend the flavour space as the orbital indices [n_hyb, n_hyb + n_int), grouped into their own
symmetry sets, and the coefficients are extended block-diagonally. hyb_coeffs must cover every fundamental operator of ad.

------

Parameters
----------
hyb_poles : {par_0}
   physical hybridization pole frequencies
hyb_coeffs : {par_1}
   hybridization coefficients at the supplied poles
tau_mesh : {par_2}
   TRIQS imaginary time DLR mesh
ad : {par_3}
   atom_diag object with field operators and invariant subspaces
dynint_ops : {par_4}
   the n_int dynamical-interaction operators
dynint_coeffs : {par_5}
   dynamical-interaction coefficients, shape (p, n_int, n_int)
)DOC",
   {{c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<triqs::mesh::dlr_imtime>()},
    {c2py::python_typename<const triqs::atom_diag::atom_diag<1> &>(), c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()},
    {c2py::python_typename<const std::vector<triqs::operators::many_body_operator_complex> &>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()}});
// compute_one_time_correlator
static auto const _c2py_fun_1cf6384f = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_efedd95d &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology"),
   c2py::cmethod(
      [](_c2py_cls_efedd95d &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_tau,
         const std::vector<triqs::operators::many_body_operator_complex> &ops_0, const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(G_ppsc, ops_tau, ops_0, ad, topology); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology")};

// compute_self_energy
static auto const _c2py_fun_e5bb193a = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_efedd95d &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

// compute_single_ptcle_gf
static auto const _c2py_fun_d1c2d042 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_efedd95d &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology); },
   "self", "G_ppsc", "topology")};

static const auto _c2py_doc_1cf6384f = _c2py_fun_1cf6384f.doc(R"DOC(
Return this MPI rank's correlator sum for one topology, without reduction.
)DOC");
static const auto _c2py_doc_e5bb193a = _c2py_fun_e5bb193a.doc(R"DOC(
Return this MPI rank's self-energy sum for one topology, without reduction.
)DOC");
static const auto _c2py_doc_d1c2d042 = _c2py_fun_d1c2d042.doc(R"DOC(
Return this MPI rank's single-particle GF sum for one topology, without reduction.
)DOC");

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_efedd95d>[] = {
   PMDF("compute_one_time_correlator", 1cf6384f),
   PMDF("compute_self_energy", e5bb193a),
   PMDF("compute_single_ptcle_gf", d1c2d042),
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

template <>
const std::string c2py::tp_doc<_c2py_cls_efedd95d> = R"DOC(Evaluate diagram contributions using block-sparse numerical storage.
Topology calls prepare this MPI rank's Backbones and discover their sector paths.
They return local sums; the caller owns reduction across ranks.
The evaluator owns persistent Action data and scratch arrays; each call supplies
the current propagator and any observable operators. Paths own their instructions
and endpoint selections, independently of discovery workspaces or result caches.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_efedd95d>;

// ==================== module functions ====================

// convolve_ppsc
static auto const _c2py_fun_83a9af88 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G1,
                 triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G2) { return triqs_xca::block_sparse::convolve_ppsc(G1, G2); },
              "G1", "G2")};

// expectation_value
static auto const _c2py_fun_14d9f3ef = c2py::dispatcher_f_kw_t{
   c2py::cfun(
      [](const triqs::operators::many_body_operator_complex &op, const triqs::atom_diag::atom_diag<0> &ad,
         triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) { return triqs_xca::block_sparse::expectation_value<false>(op, ad, G_ppsc); },
      "op", "ad", "G_ppsc"),
   c2py::cfun(
      [](const triqs::operators::many_body_operator_complex &op, const triqs::atom_diag::atom_diag<1> &ad,
         triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) { return triqs_xca::block_sparse::expectation_value<true>(op, ad, G_ppsc); },
      "op", "ad", "G_ppsc")};

// sigma_nca
static auto const _c2py_fun_c08d5311 =
   c2py::dispatcher_f_kw_t{c2py::cfun([](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            hyb,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            hyb_refl,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            Gt,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            Fs,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            F_dags) { return triqs_xca::dense::manual::sigma_nca(hyb, hyb_refl, Gt, Fs, F_dags); },
                                      "hyb", "hyb_refl", "Gt", "Fs", "F_dags")};

// sigma_oca
static auto const _c2py_fun_71c5744e = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    hyb,
                 cppdlr::imtime_ops itops, double beta,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    Gt,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    Fs,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    F_dags) { return triqs_xca::dense::manual::sigma_oca(hyb, itops, beta, Gt, Fs, F_dags); },
              "hyb", "itops", "beta", "Gt", "Fs", "F_dags"),
   c2py::cfun(
      [](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_coeffs,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_refl,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_refl_coeffs,
         nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_poles,
         cppdlr::imtime_ops &itops, double beta,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            Gt,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            Fs,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            F_dags) {
        return triqs_xca::dense::manual::sigma_oca(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt, Fs, F_dags);
      },
      "hyb", "hyb_coeffs", "hyb_refl", "hyb_refl_coeffs", "hyb_poles", "itops", "beta", "Gt", "Fs", "F_dags")};

// trace
static auto const _c2py_fun_813d75ae =
   c2py::dispatcher_f_kw_t{c2py::cfun([](triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) { return triqs_xca::block_sparse::trace(G); }, "G")};

static const auto _c2py_doc_83a9af88 = _c2py_fun_83a9af88.doc(R"DOC(
Compute the Volterra "convolution" of two pseudo-particle Green's functions

Parameters
----------
G1 : {par_0}
   Left Green's function
G2 : {par_1}
   Right Green's function

Returns
-------
{ret_0}
   Convolution (G1 * G2)()
)DOC",
                                                              {{c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()},
                                                               {c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}},
                                                              {c2py::python_typename<triqs::gfs::block_gf<triqs::mesh::dlr_imtime>>()});
static const auto _c2py_doc_14d9f3ef = _c2py_fun_14d9f3ef.doc(
   R"DOC(
Compute the expectation value of the 2nd quantized operator op, <O> = -Tr[O G()]
using the AtomDiag instance ad to generate a block representation
and tracing with the many-body density matrix of the pseudo particle Green's function G_ppsc

Parameters
----------
op : {par_0}
   2nd quantization operator
ad : {par_1}
   AtomDiag object
G_ppsc : {par_2}
   Pseudo-particle Green's function

Returns
-------
{ret_0}
   Expectation value -Tr[G() O]
)DOC",
   {{c2py::python_typename<const triqs::operators::many_body_operator_complex &>()},
    {c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>(), c2py::python_typename<const triqs::atom_diag::atom_diag<1> &>()},
    {c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}},
   {c2py::python_typename<nda::dcomplex>()});
static const auto _c2py_doc_c08d5311 =
   _c2py_fun_c08d5311.doc(R"DOC(
Evaluate NCA self-energy term using dense storage

Parameters
----------
hyb : {par_0}
   hybridization function
hyb_refl : {par_1}
   hyb evaluated at (beta - tau)
Gt : {par_2}
   Greens function
Fs : {par_3}
   vector of annihilation operators
F_dags : {par_4}
   vector of creation operators
)DOC",
                          {{c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A',
                                                                        nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                           {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A',
                                                                        nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                           {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A',
                                                                        nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                           {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A',
                                                                        nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                           {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A',
                                                                        nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()}});
static const auto _c2py_doc_71c5744e = _c2py_fun_71c5744e.doc(
   R"DOC(
[1] Evaluate OCA using dense storage

------

[2] Evaluate OCA using dense storage and allow user to provide hybridization poles and coefficients

------

Parameters
----------
hyb : {par_0}
   hybridization function at imaginary time nodes
itops : {par_1}
   cppdlr imaginary time object
beta : {par_2}
   inverse temperature
Gt : {par_3}
   Greens function
Fs : {par_4}
   F operator
hyb_coeffs : {par_5}
   hybridization coefficients
hyb_refl : {par_6}
   hybridization function eval'd at negative imag. times
hyb_refl_coeffs : {par_7}
   hybridization coefficients at negative imag. times
hyb_poles : {par_8}
   hybridization poles in physical frequency units (not beta-scaled)

Returns
-------
{ret_0}
   OCA term of self-energy
)DOC",
   {{c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<cppdlr::imtime_ops>(), c2py::python_typename<cppdlr::imtime_ops &>()},
    {c2py::python_typename<double>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()}},
   {c2py::python_typename<
      nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});
static const auto _c2py_doc_813d75ae = _c2py_fun_813d75ae.doc(R"DOC(
Take the trace of a pseudo-particle Green's function

Parameters
----------
G : {par_0}
   Pseudo-particle Green's function
)DOC",
                                                              {{c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}});
//--------------------- module function table  -----------------------------

// clang-format off
static PyMethodDef module_methods[] = {
   PMDF("convolve_ppsc", 83a9af88),
   PMDF("expectation_value", 14d9f3ef),
   PMDF("sigma_nca", c08d5311),
   PMDF("sigma_oca", 71c5744e),
   PMDF("trace", 813d75ae),
   {nullptr, nullptr, 0, nullptr}  // Sentinel
};
// clang-format on

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "_triqs_xca",      /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit__triqs_xca() {

  if (not c2py::check_python_version("_triqs_xca")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_698c8c52>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_f8dfe6a1>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_efedd95d>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  if (not c2py::register_internal_types()) return NULL;
#define _add_type(T, N)                                                                                                                              \
  if (not c2py::add_type_object_to_main<T>(N, m)) return NULL
  _add_type(_c2py_cls_698c8c52, "DenseDiagramEvaluator");
  _add_type(_c2py_cls_f8dfe6a1, "DenseDiagramEvaluatorFewOrb");
  _add_type(_c2py_cls_efedd95d, "BlockSparseDiagramEvaluator");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
