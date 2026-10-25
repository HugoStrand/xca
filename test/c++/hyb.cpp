#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/action.hpp"
#include "triqs_xca/dense/action.hpp"
#include <gtest/gtest.h>

#include <nda/nda.hpp>
#include <triqs/gfs.hpp>

#include <cppdlr/dlr_kernels.hpp>

using triqs_xca::diagram::Direction;

/**
 * @file hyb.cpp
 *
 * @brief Tests of Action scalar data and tau-kernel binding
 *
 * @details The coefficients reach the library as a generic nda view, so a view with any strides has to give the same hybridization as a
 * contiguous copy of the same values.
 */

using nda::dcomplex;
using nda::range;

/// @brief Views of the coefficients with a skipped column and with a reversed pole index give the same hybridization as contiguous copies
TEST(Hybridization, strided_coefficients) {
  double beta = 2.0, Lambda = 40.0, eps = 1e-10;
  auto tau_mesh             = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
  nda::vector<double> poles = {1.3, -0.8};
  long p = poles.size(), n = 3;

  // every other column of a wider array, with filler values in the skipped columns
  nda::array<dcomplex, 3> wide(p, n, 2 * n);
  wide = dcomplex(99.0, -7.0);
  for (long l = 0; l < p; ++l)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) wide(l, i, 2 * j) = dcomplex(1.0 + l + i, 0.5 * j - i);
  auto strided = wide(range::all, range::all, range(0, 2 * n, 2));
  ASSERT_FALSE(strided.indexmap().is_contiguous());

  // a negative stride, which nda reports as contiguous
  nda::array<dcomplex, 3> base = strided;
  auto reversed                = base(range(p - 1, -1, -1), range::all, range::all);
  ASSERT_TRUE(reversed.indexmap().is_contiguous());

  auto zero_operators = nda::zeros<dcomplex>(n, 1, 1);
  triqs_xca::dense::FSet operators(zero_operators, zero_operators, nda::zeros<dcomplex>(p, n, n));
  for (auto const &view : {nda::array_const_view<dcomplex, 3>(strided), nda::array_const_view<dcomplex, 3>(reversed)}) {
    nda::array<dcomplex, 3> contiguous = view;
    auto H_view                        = triqs_xca::dense::make_action(tau_mesh, poles, view, operators);
    auto H_contiguous                  = triqs_xca::dense::make_action(tau_mesh, poles, contiguous, operators);
    ASSERT_GT(nda::max_element(nda::abs(H_contiguous.tau_values)), 0.1);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.pole_coeffs - H_contiguous.pole_coeffs)), 0.0);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.tau_values - H_contiguous.tau_values)), 0.0);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.tau_values_reflect - H_contiguous.tau_values_reflect)), 0.0);
  }
}


TEST(Hybridization, vertex_kernels_include_reflection_and_identity) {
  using namespace triqs_xca;
  double beta = 2.;
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{-0.8};
  auto kernel = make_tau_kernel(mesh, poles);
  auto diagram = diagram::prepare_diagram(diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 1,
                                          diagram::Arcs{{Direction::forward, -1, -1}, {Direction::forward, 0, 0}}, poles, diagram::Kind::self_energy);
  auto nodes = mesh.dlr_it().get_itnodes();
  for (bool reflected : {false, true}) {
    for (int vertex = 0; vertex < 4; ++vertex) {
      auto values = nda::ones<dcomplex>(nodes.size(), 1, 1);
      apply_kernel(values, diagram.backbone.vertices[vertex], diagram.arcs, kernel, reflected);
      for (int t = 0; t < nodes.size(); ++t) {
        // Forward internal arc, negative pole: earlier vertex K(-pole), later K(+pole).
        double argument = (vertex == 1 ? 0.8 : -0.8) * beta * (reflected ? -1 : 1);
        dcomplex expected = vertex == 0 || vertex == 2 ? 1. : cppdlr::k_it(nodes(t), argument);
        EXPECT_EQ(values(t, 0, 0), expected);
      }
    }
  }
  EXPECT_DOUBLE_EQ(tau_kernel_prefactor(diagram.backbone.kernel_prefactor, diagram.arcs, kernel),
                   std::pow(cppdlr::k_it(0., 0.8 * beta), -1));
  for (int edge = 0; edge < 3; ++edge) {
    auto values = nda::ones<dcomplex>(nodes.size(), 1, 1);
    apply_kernel_product(values, diagram.backbone.edges[edge].kernels, diagram.arcs, kernel);
    EXPECT_EQ(nda::max_element(nda::abs(values - dcomplex(1))), 0.);
  }
}

TEST(Hybridization, edge_products_read_the_supplied_tables) {
  using namespace triqs_xca;
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{0.7, 1.3};
  auto kernel = make_tau_kernel(mesh, poles);
  auto diagram = diagram::prepare_diagram(diagram::Topology{nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}, 1,
                                          diagram::Arcs{{Direction::forward, -1, -1}, {Direction::forward, 0, 0}, {Direction::forward, 0, 1}}, poles, diagram::Kind::self_energy);
  auto nodes = mesh.dlr_it().get_itnodes();
  auto values = nda::ones<dcomplex>(nodes.size(), 1, 1);
  auto edge = diagram.backbone.edges[2].kernels;
  apply_kernel_product(values, edge, diagram.arcs, kernel);
  for (int t = 0; t < nodes.size(); ++t)
    EXPECT_EQ(values(t, 0, 0), dcomplex(cppdlr::k_it(nodes(t), 1.4) * cppdlr::k_it(nodes(t), 2.6)));
  EXPECT_DOUBLE_EQ(tau_kernel_prefactor(diagram.backbone.kernel_prefactor, diagram.arcs, kernel),
                   std::pow(cppdlr::k_it(0., 1.4), -2) * std::pow(cppdlr::k_it(0., 2.6), -2));

  // The recipe remains valid when pole magnitudes change without changing signs.
  auto changed = make_tau_kernel(mesh, nda::vector<double>{0.9, 1.5});
  values = 1;
  apply_kernel_product(values, edge, diagram.arcs, changed);
  for (int t = 0; t < nodes.size(); ++t)
    EXPECT_EQ(values(t, 0, 0), dcomplex(cppdlr::k_it(nodes(t), 1.8) * cppdlr::k_it(nodes(t), 3.0)));
}

TEST(Hybridization, edge_factors_preserve_slot_order_and_skip_identity_slots) {
  using namespace triqs_xca;
  // Reordering the active factors overflows; combining them before multiplication
  // underflows in the second case. Keep the existing sequential arithmetic.
  TauKernel kernel{nda::vector<double>(2), nda::vector<double>(2),
                         nda::array<double, 2>(1, 2), nda::array<double, 2>(1, 2)};
  kernel.at_tau_plus = 1e200;
  kernel.at_tau_minus = 1e200;
  kernel.at_tau_plus(0, 1) = 1e-200;
  nda::array<dcomplex, 3> values(1, 1, 1);
  values = 1e200;
  apply_kernel_product(values, std::vector<diagram::KernelFactor>{{1, 1}, {3, -1}}, diagram::Arcs{{}, {diagram::Direction::forward, 0, 1}, {diagram::Direction::forward, 0, 0}, {diagram::Direction::forward, 0, 1}}, kernel);
  EXPECT_DOUBLE_EQ(values(0, 0, 0).real(), 1e200);
  EXPECT_DOUBLE_EQ(values(0, 0, 0).imag(), 0.);
  values = 1e200;
  apply_kernel_product(values, std::vector<diagram::KernelFactor>{{1, 1}, {2, 1}}, diagram::Arcs{{}, {diagram::Direction::forward, 0, 1}, {diagram::Direction::forward, 0, 1}}, kernel);
  EXPECT_DOUBLE_EQ(values(0, 0, 0).real(), 1e-200);
}

TEST(Hybridization, trailing_correlator_edge_is_identity) {
  using namespace triqs_xca;
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{0.7};
  auto data = make_tau_kernel(mesh, poles);
  for (auto topology_arr : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
    diagram::Topology topology{topology_arr};
    auto diagram = diagram::prepare_diagram(topology, 1, poles, 0, diagram::Kind::correlator);
    auto const &edges = diagram.backbone.edges;
    ASSERT_EQ(edges.size(), 2 * topology.order());
    auto values = nda::ones<dcomplex>(data.at_tau_plus.extent(0), 1, 1);
    apply_kernel_product(values, edges.back().kernels, diagram.arcs, data);
    EXPECT_EQ(nda::max_element(nda::abs(values - dcomplex(1))), 0.);
    if (topology.order() == 1) {
      EXPECT_DOUBLE_EQ(tau_kernel_prefactor(diagram.backbone.kernel_prefactor, diagram.arcs, data), 1.);
      apply_kernel(values, diagram.backbone.vertices[0], diagram.arcs, data);
      EXPECT_EQ(nda::max_element(nda::abs(values - dcomplex(1))), 0.);
    }
  }
}
