#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/tau_kernel_ops.hpp"
#include "triqs_xca/dense/action.hpp"
#include <gtest/gtest.h>

#include <nda/nda.hpp>
#include <triqs/gfs.hpp>

#include <triqs_xca/hyb.hpp>
#include <triqs_xca/bind/tau_kernel.hpp>
#include <cppdlr/dlr_kernels.hpp>

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


TEST(Hybridization, bound_vertex_kernels_include_reflection_and_identity) {
  using namespace triqs_xca;
  double beta = 2.;
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{-0.8};
  auto kernel = make_tau_kernel_values(mesh, poles);
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 1,
                                       triqs_xca::diagram::Arcs{{1, 1}, {0}, {0}}, poles, triqs_xca::diagram::Kind::self_energy);
  auto bound = triqs_xca::bind::tau_kernels(assignment.backbone.kernel_placement, assignment.kernel_prefactor, assignment.arcs.pole_indices, kernel);
  ASSERT_EQ(bound.vertices.size(), 4);
  EXPECT_FALSE(bound.vertices[0]);
  EXPECT_FALSE(bound.vertices[2]);
  ASSERT_TRUE(bound.vertices[1]);
  ASSERT_TRUE(bound.vertices[3]);
  ASSERT_TRUE(bound.reflected_vertices[1]);
  ASSERT_TRUE(bound.reflected_vertices[3]);
  EXPECT_EQ(bound.vertices[1]->data(), kernel.at_tau_minus(range::all, 0).data());
  auto nodes = mesh.dlr_it().get_itnodes();
  // Forward internal line with a negative pole: K(-pole) on the earlier
  // vertex and K(+pole) on the later one. Reflection exchanges those tables.
  for (int t = 0; t < nodes.size(); ++t) {
    EXPECT_DOUBLE_EQ((*bound.vertices[1])(t), cppdlr::k_it(nodes(t), 0.8 * beta));
    EXPECT_DOUBLE_EQ((*bound.vertices[3])(t), cppdlr::k_it(nodes(t), -0.8 * beta));
    EXPECT_DOUBLE_EQ((*bound.reflected_vertices[1])(t), cppdlr::k_it(nodes(t), -0.8 * beta));
    EXPECT_DOUBLE_EQ((*bound.reflected_vertices[3])(t), cppdlr::k_it(nodes(t), 0.8 * beta));
  }
  EXPECT_DOUBLE_EQ(bound.prefactor, std::pow(cppdlr::k_it(0., 0.8 * beta), -1));
  for (auto const &edge : bound.edges) EXPECT_TRUE(edge.empty());
  auto values = nda::ones<dcomplex>(nodes.size(), 1, 1);
  apply_kernel(values, bound.vertices[0]);
  EXPECT_EQ(nda::max_element(nda::abs(values - dcomplex(1))), 0.);
  apply_kernel(values, bound.reflected_vertices[1]);
  for (int t = 0; t < nodes.size(); ++t) EXPECT_EQ(values(t, 0, 0), dcomplex(cppdlr::k_it(nodes(t), -0.8 * beta)));
}

TEST(Hybridization, bound_edge_products_preserve_factors_and_rebind_to_new_tables) {
  using namespace triqs_xca;
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{0.7, 1.3};
  auto kernel = make_tau_kernel_values(mesh, poles);
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}, 1,
                                       triqs_xca::diagram::Arcs{{1, 1, 1}, {0, 1}, {0, 0}}, poles, triqs_xca::diagram::Kind::self_energy);
  auto bound = triqs_xca::bind::tau_kernels(assignment.backbone.kernel_placement, assignment.kernel_prefactor, assignment.arcs.pole_indices, kernel);
  ASSERT_EQ(bound.edges.size(), 5);
  EXPECT_TRUE(bound.edges[0].empty());
  EXPECT_EQ(bound.edges[1].size(), 1);
  ASSERT_EQ(bound.edges[2].size(), 2);
  EXPECT_EQ(bound.edges[3].size(), 2);
  EXPECT_EQ(bound.edges[4].size(), 1);
  EXPECT_EQ(bound.edges[2][0].data(), kernel.at_tau_plus(range::all, 0).data());
  EXPECT_EQ(bound.edges[2][1].data(), kernel.at_tau_plus(range::all, 1).data());
  auto nodes = mesh.dlr_it().get_itnodes();
  auto values = nda::ones<dcomplex>(nodes.size(), 1, 1);
  apply_kernel_product(values, bound.edges[2]);
  for (int t = 0; t < nodes.size(); ++t)
    EXPECT_EQ(values(t, 0, 0), dcomplex(cppdlr::k_it(nodes(t), 1.4) * cppdlr::k_it(nodes(t), 2.6)));
  EXPECT_DOUBLE_EQ(bound.prefactor, std::pow(cppdlr::k_it(0., 1.4), -2) * std::pow(cppdlr::k_it(0., 2.6), -2));

  // The recipe stays valid when positive poles change magnitude, but numerical
  // operands must be rebound. The old binding still refers to its original owner.
  auto changed = make_tau_kernel_values(mesh, nda::vector<double>{0.9, 1.5});
  auto rebound = triqs_xca::bind::tau_kernels(assignment.backbone.kernel_placement, assignment.kernel_prefactor, assignment.arcs.pole_indices, changed);
  EXPECT_NE(rebound.edges[2][0].data(), bound.edges[2][0].data());
  EXPECT_EQ(rebound.edges[2][0].data(), changed.at_tau_plus(range::all, 0).data());
  EXPECT_DOUBLE_EQ(rebound.edges[2][0](0), cppdlr::k_it(nodes(0), 1.8));
  EXPECT_DOUBLE_EQ(bound.edges[2][0](0), cppdlr::k_it(nodes(0), 1.4));
}

TEST(Hybridization, trailing_correlator_edge_binds_as_identity) {
  using namespace triqs_xca;
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{0.7};
  auto data = make_tau_kernel_values(mesh, poles);
  for (auto topology_arr : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
    diagram::Topology topology{topology_arr};
    auto diagram = diagram::prepare_diagram(topology, 1, poles, 0, diagram::Kind::correlator);
    auto kernels = bind::tau_kernels(diagram.backbone.kernel_placement, diagram.kernel_prefactor, diagram.arcs.pole_indices, data);
    ASSERT_EQ(kernels.edges.size(), 2 * topology.order());
    EXPECT_TRUE(kernels.edges.back().empty());
    auto values = nda::ones<dcomplex>(data.at_tau_plus.extent(0), 1, 1);
    apply_kernel_product(values, kernels.edges.back());
    EXPECT_EQ(nda::max_element(nda::abs(values - dcomplex(1))), 0.);
  }
}
