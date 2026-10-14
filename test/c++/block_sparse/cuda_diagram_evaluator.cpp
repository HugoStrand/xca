#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/hyb.hpp>
#include <limits>
#include <cstdlib>
#include "test_utils/block_sparse.hpp"
#include "test_utils/correlator_precision.hpp"
#include <nda/lapack.hpp>

namespace {
  namespace bs  = triqs_xca::block_sparse;
  namespace gpu = bs::cuda;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  constexpr auto all = nda::range::all;
  auto atom(int partition) {
    triqs::atom_diag::fundamental_operator_set fops;
    many_body_operator_complex H, Q;
    for (int i = 0; i < 3; ++i) {
      fops.insert("f", i);
      Q += n("f", i);
      H += (.13 + .07 * i) * n("f", i);
      for (int j = 0; j < i; ++j) {
        H += .2 * n("f", i) * n("f", j);
        if (partition == 1 || (partition == 2 && i == 1)) {
          dcomplex z{.03 * (i + j + 1), .017 * (i - j)};
          H += z * c_dag("f", i) * c("f", j) + std::conj(z) * c_dag("f", j) * c("f", i);
        }
      }
    }
    if (partition == 0) return triqs::atom_diag::atom_diag<true>(H, fops);
    std::vector<many_body_operator_complex> sym =
       partition == 1 ? std::vector<many_body_operator_complex>{Q} : std::vector<many_body_operator_complex>{n("f", 0) + n("f", 1), n("f", 2)};
    return triqs::atom_diag::atom_diag<true>(H, fops, sym);
  }
  bs::BlockOpSymQuartet direct_quartet(gpu::OperatorModel const &model) {
    std::vector<bs::BlockOpSymSet> Fs, Fd;
    auto nb = model.dimensions().size();
    for (auto const &group : model.sets()) {
      for (bool dag : {false, true}) {
        nda::vector<int> connections(nb);
        std::vector<nda::array<dcomplex, 3>> blocks(nb);
        for (std::size_t b = 0; b < nb; ++b) {
          auto const &first = model.field(group[0], dag);
          connections(b)    = first.connection[b];
          if (connections(b) == -1)
            blocks[b] = nda::zeros<dcomplex>(group.size(), 1, 1);
          else {
            blocks[b] = nda::zeros<dcomplex>(group.size(), first.blocks[b].rows, first.blocks[b].columns);
            for (std::size_t k = 0; k < group.size(); ++k) blocks[b](k, all, all) = model.field(group[k], dag).blocks[b].to_dense();
          }
        }
        (dag ? Fd : Fs).emplace_back(connections, blocks);
      }
    }
    nda::vector<long> labels(model.num_fields());
    for (int i = 0; i < model.num_fields(); ++i) labels(i) = model.labels()[i];
    return {Fs, Fd, model.coefficients(), labels};
  }
  double compare(auto const &actual, auto const &expected, double tolerance = 1e-13) {
    EXPECT_EQ(actual.size(), expected.size());
    double signal = 0;
    for (long b = 0; b < expected.size(); ++b) {
      auto scale = nda::max_element(nda::abs(expected[b].data()));
      signal     = std::max(signal, scale);
      EXPECT_LE(nda::max_element(nda::abs(actual[b].data() - expected[b].data())), tolerance * std::max(scale, 1e-20)) << "block " << b;
    }
    return signal;
  }
  bs::BlockOp endpoint(gpu::SparseOperator const &op) {
    nda::vector<int> connection(op.connection.size());
    std::vector<nda::array<dcomplex, 2>> blocks;
    for (std::size_t b = 0; b < op.connection.size(); ++b) {
      connection(b) = op.connection[b];
      if (op.connection[b] == -1)
        blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      else
        blocks.emplace_back(op.blocks[b].to_dense());
    }
    return {connection, blocks};
  }
  nda::array<dcomplex, 3> cpu_correlator(bs::DiagramEvaluator &cpu, auto const &G, gpu::OperatorModel const &model,
                                         std::vector<gpu::SparseOperator> const &mu, std::vector<gpu::SparseOperator> const &kap, bool fermionic,
                                         nda::array_const_view<int, 2> topology, long first, long last) {
    bs::BlockDiagOpFun input(G);
    CorrelatorBackbone backbone(topology, model.num_fields(), model.num_interactions());
    std::vector<bs::BlockOp> m, k;
    for (auto const &op : mu) m.push_back(endpoint(op));
    for (auto const &op : kap) k.push_back(endpoint(op));
    auto answer = nda::zeros<dcomplex>(G[0].mesh().dlr_it().rank(), mu.size(), kap.size());
    for (long index = first; index < last; ++index) answer += cpu.eval_correlator(input, backbone, m, k, index, fermionic);
    return answer;
  }
  void compare_array(nda::array_const_view<dcomplex, 3> actual, nda::array_const_view<dcomplex, 3> expected, double tolerance = 1e-13) {
    ASSERT_EQ(actual.shape(), expected.shape());
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), tolerance * std::max(1e-20, nda::max_element(nda::abs(expected))));
  }
} // namespace
TEST(BlockSparseCudaEvaluator, SpgfAndBosonicCorrelatorExactInputs) {
  for (int partition : {0, 1, 2}) {
    SCOPED_TRACE(partition);
    auto ad                = atom(partition);
    auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
    auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
    std::vector<many_body_operator_complex> bosons{n("f", 0) + n("f", 1) + n("f", 2), c_dag("f", 0) * c("f", 1)};
    auto dyn     = nda::zeros<dcomplex>(2, 2, 2);
    dyn(0, 0, 0) = .07;
    dyn(1, 1, 1) = {.03, .01};
    auto model   = gpu::export_model(ad, coeff, bosons, dyn);
    auto quartet = direct_quartet(model);
    nda::vector<double> poles{1.3, -.8};
    auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
    bs::DiagramEvaluator cpu(2., 40., 1e-10, poles, model.coefficients(), quartet, bosons.size());
    bs::DiagramEvaluator rotated(poles, coeff, G[0].mesh(), ad, bosons, dyn);
    gpu::DiagramEvaluator device(model, poles, G[0].mesh());
    std::vector<gpu::SparseOperator> mu, kap;
    for (auto const &op : bosons) {
      mu.push_back(gpu::export_operator(ad, op));
      kap.push_back(gpu::export_operator(ad, dagger(op)));
    }
    for (auto topology :
         {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 1}, {2, 3}}, nda::array<int, 2>{{0, 3}, {1, 2}},
          nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}, nda::array<int, 2>{{0, 4}, {1, 5}, {2, 6}, {3, 7}}}) {
      long total = device.get_num_single_ptcle_gf_backbones(topology);
      EXPECT_EQ(total, cpu.get_num_single_ptcle_gf_backbones(topology));
      for (long first : {0L, total / 2, std::max(0L, total - 2)}) {
        SCOPED_TRACE(::testing::Message() << "order " << topology.extent(0) << " special " << topology(0, 1) << " index " << first);
        long last     = std::min(total, first + 1);
        auto expected = cpu.compute_single_ptcle_gf_range(G, topology, first, last);
        {
          SCOPED_TRACE("SPGF CUDA versus exact CPU");
          compare_array(device.compute_single_ptcle_gf_range(G, topology, first, last), expected);
        }
        auto raw = cpu_correlator(cpu, G, model, mu, kap, false, topology, first, last);
        {
          SCOPED_TRACE("bosonic CUDA versus exact CPU");
          auto actual = device.compute_correlator_range(G, mu, kap, false, topology, first, last);
          if (topology.extent(0) >= 4) {
            auto precise = triqs_xca::test_utils::precision::correlator(model, G, poles, mu, kap, false, topology, first);
            // Independently accumulated arithmetic exposes the legacy CPU's
            // high-order rounding floor; qualify each result against it too.
            compare_array(raw, precise, 1e-12);
            compare_array(actual, precise, 1e-12);
          }
          compare_array(actual, raw, topology.extent(0) < 4 ? 1e-13 : 1e-12);
        }
        auto counts = device.last_path_counts();
        EXPECT_EQ(counts.proposed, counts.accepted + counts.pruned);
        if (auto *isolated = std::getenv("XCA_BS_CHECK_MEMORY_PEAK"); isolated && std::string(isolated) == "1")
          EXPECT_LE(device.last_device_peak_delta(), device.correlator_storage_bound(mu.size(), kap.size()) + 64ULL * 1024 * 1024);
      }
    }
    nda::array<int, 2> nca{{0, 1}};
    compare_array(device.compute_single_ptcle_gf(G, nca), rotated.compute_single_ptcle_gf(G, nca));
    auto changed      = G;
    changed[1].data() = 0.;
    compare_array(device.compute_single_ptcle_gf(changed, nca), cpu.compute_single_ptcle_gf(changed, nca));
    EXPECT_THROW(device.compute_single_ptcle_gf(G, nca, 1), std::out_of_range);
    auto empty = device.compute_single_ptcle_gf_range(G, nca, 1, 1);
    EXPECT_EQ(nda::max_element(nda::abs(empty)), 0.);
    std::vector<many_body_operator_complex> daggered;
    for (auto const &op : bosons) daggered.push_back(dagger(op));
    compare_array(device.compute_one_time_correlator(G, bosons, daggered, ad, nca),
                  rotated.compute_one_time_correlator_range(G, bosons, daggered, ad, nca, 0, 1));
    bosons[1] *= .37; // Same temporary allocation may be reused with new values.
    compare_array(device.compute_one_time_correlator(G, bosons, daggered, ad, nca, 0),
                  rotated.compute_one_time_correlator_range(G, bosons, daggered, ad, nca, 0, 1));
  }
}
TEST(BlockSparseCudaEvaluator, CorrelatorApiErrorsRecoveryAndRangeAdditivity) {
  auto ad                = atom(1);
  auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  auto model             = gpu::export_model(ad, coeff);
  auto quartet           = direct_quartet(model);
  nda::vector<double> poles{1.3, -.8};
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  gpu::DiagramEvaluator device(model, poles, G[0].mesh()), limited(model, poles, G[0].mesh(), 65536);
  bs::DiagramEvaluator cpu(2., 40., 1e-10, poles, coeff, quartet);
  nda::array<int, 2> oca{{0, 2}, {1, 3}}, nca{{0, 1}};
  long total = device.get_num_single_ptcle_gf_backbones(oca);
  auto full = device.compute_single_ptcle_gf(G, oca), first = device.compute_single_ptcle_gf_range(G, oca, 0, total / 2),
       last = device.compute_single_ptcle_gf_range(G, oca, total / 2, total);
  compare_array(nda::array<dcomplex, 3>(first + last), full);
  compare_array(full, cpu.compute_single_ptcle_gf(G, oca));
  EXPECT_THROW(limited.compute_single_ptcle_gf(G, nca), std::length_error);
  EXPECT_THROW(device.compute_single_ptcle_gf_range(G, nca, -1, 0), std::out_of_range);
  EXPECT_THROW(device.compute_single_ptcle_gf(G, nca, 1), std::out_of_range);
  std::vector<many_body_operator_complex> mixed{c("f", 0), n("f", 0)}, single{c_dag("f", 0)}, zero{many_body_operator_complex{}}, none;
  EXPECT_THROW(device.compute_one_time_correlator(G, mixed, single, ad, nca), std::runtime_error);
  EXPECT_THROW(device.compute_one_time_correlator(G, zero, zero, ad, nca), std::runtime_error);
  EXPECT_THROW(device.compute_one_time_correlator(G, none, none, ad, nca), std::runtime_error);
  std::vector<gpu::SparseOperator> mu{model.field(0)}, kap{model.field(0, true)}, empty;
  auto result = device.compute_correlator_range(G, empty, kap, true, nca, 0, 1);
  EXPECT_EQ(result.shape(), (std::array<long, 3>{G[0].mesh().dlr_it().rank(), 0, 1}));
  gpu::SparseOperator wrong;
  wrong.dimensions = {1};
  wrong.connection = {-1};
  wrong.blocks     = {gpu::SparseBlock{0, 1, {0}, {}, {}}};
  std::vector<gpu::SparseOperator> bad{wrong};
  EXPECT_THROW(device.compute_correlator_range(G, bad, kap, true, nca, 0, 1), std::invalid_argument);
  compare_array(device.compute_single_ptcle_gf(G, nca), cpu.compute_single_ptcle_gf(G, nca));
  compare(device.compute_self_energy(G, nca), cpu.compute_self_energy(G, nca));
}
TEST(BlockSparseCudaEvaluator, RealAtomDiagFermionicEndpointOverload) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("f", 0);
  triqs::atom_diag::atom_diag<false> ad(.2 * n("f", 0), fops);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto coeff     = nda::zeros<dcomplex>(2, 1, 1);
  coeff(0, 0, 0) = .3;
  coeff(1, 0, 0) = .17;
  nda::vector<double> poles{1.3, -.8};
  auto model = gpu::export_model(ad, coeff);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks;
  auto nodes = mesh.dlr_it().get_itnodes();
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto values = nda::zeros<dcomplex>(mesh.dlr_it().rank(), 1, 1);
    for (long t = 0; t < values.extent(0); ++t)
      values(t, 0, 0) = -std::exp(-2. * (nodes(t) > 0 ? nodes(t) : 1. + nodes(t)) * ad.get_eigensystems()[b].eigenvalues(0));
    blocks.emplace_back(mesh, values);
  }
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G{blocks};
  gpu::DiagramEvaluator device(model, poles, mesh);
  bs::DiagramEvaluator cpu(poles, coeff, mesh, ad);
  std::vector<many_body_operator_complex> mu{c("f", 0)}, kap{c_dag("f", 0)};
  nda::array<int, 2> nca{{0, 1}};
  compare_array(device.compute_one_time_correlator_range(G, mu, kap, ad, nca, 0, 1),
                cpu.compute_one_time_correlator_range(G, mu, kap, ad, nca, 0, 1));
  compare_array(device.compute_single_ptcle_gf(G, nca), cpu.compute_single_ptcle_gf(G, nca));
}
TEST(BlockSparseCudaEvaluator, LongDoubleCachedSolveAndConvolutionControl) {
  nda::vector<double> rf{-2., 0., 2.}, nodes{0., .4, -.1};
  nda::matrix<double> C(3, 3);
  for (long i = 0; i < 3; ++i)
    for (long j = 0; j < 3; ++j) C(i, j) = cppdlr::k_it(nodes(i), rf(j));
  auto lu = C;
  nda::vector<int> piv(3);
  ASSERT_EQ(nda::lapack::getrf(lu, piv), 0);
  cppdlr::imtime_ops host(4., rf, nodes, C, lu, piv);
  triqs_xca::test_utils::precision::Dlr extended(host);
  auto fc = nda::zeros<dcomplex>(3, 2, 3), gc = nda::zeros<dcomplex>(3, 3, 2);
  for (long t = 0; t < 3; ++t)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 3; ++j) fc(t, i, j) = {.1 + .03 * t + .02 * i - .01 * j, .02 * t - .05 * i + .03 * j};
  for (long t = 0; t < 3; ++t)
    for (long i = 0; i < 3; ++i)
      for (long j = 0; j < 2; ++j) gc(t, i, j) = {.2 - .04 * t + .03 * i + .05 * j, -.01 * t + .02 * i - .06 * j};
  auto values = host.coefs2vals(fc);
  auto solved = extended.solve(triqs_xca::test_utils::precision::Function(values));
  for (long t = 0; t < 3; ++t)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 3; ++j) EXPECT_LE(std::abs(solved(t, i, j) - triqs_xca::test_utils::precision::extended(fc(t, i, j))), 1e-12L);
  auto answer   = extended.convolve(2., triqs_xca::test_utils::precision::Function(fc), triqs_xca::test_utils::precision::Function(gc));
  auto expected = host.convolve(2., fc, gc, cppdlr::TIME_ORDERED);
  for (long t = 0; t < 3; ++t)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 2; ++j) EXPECT_LE(std::abs(answer(t, i, j) - triqs_xca::test_utils::precision::extended(expected(t, i, j))), 1e-12L);
}
TEST(BlockSparseCudaEvaluator, CorrelatorKeepsMaskedStartingValues) {
  auto ad    = atom(0);
  auto coeff = nda::zeros<dcomplex>(2, 3, 3);
  for (int i = 0; i < 3; ++i) {
    coeff(0, i, i) = .3;
    coeff(1, i, i) = .17;
  }
  auto model   = gpu::export_model(ad, coeff);
  auto quartet = direct_quartet(model);
  nda::vector<double> poles{1.3, -.8};
  auto G   = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  int tiny = -1, other = -1;
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    if (ad.get_fock_states(b)[0] == 7) tiny = b;
    if (ad.get_fock_states(b)[0] == 3) other = b;
  }
  ASSERT_GE(tiny, 0);
  ASSERT_GE(other, 0);
  G[tiny].data() *= 1e-18;
  G[other].data() = 0.;
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  CorrelatorBackbone bb(topology, 3);
  long index = -1;
  for (long i = 0; i < bb.num_flat_indices(2); ++i) {
    bb.set_flat_index(i, poles);
    if (bb.get_vertex_orb(1) == 0 && !bb.has_vertex_dag(1)) {
      index = i;
      break;
    }
  }
  ASSERT_GE(index, 0);
  bs::DiagramEvaluator cpu(2., 40., 1e-10, poles, coeff, quartet);
  gpu::DiagramEvaluator device(model, poles, G[0].mesh());
  auto gold = cpu.compute_single_ptcle_gf(G, topology, index), actual = device.compute_single_ptcle_gf(G, topology, index);
  double scale = nda::max_element(nda::abs(gold(all, 1, 1)));
  EXPECT_GT(scale, 1e-24);
  EXPECT_LE(nda::max_element(nda::abs(actual(all, 1, 1) - gold(all, 1, 1))), 1e-13 * scale);
  G[tiny].data() = 0.;
  actual         = device.compute_single_ptcle_gf(G, topology, index);
  EXPECT_EQ(nda::max_element(nda::abs(actual(all, 1, 1))), 0.);
}
TEST(BlockSparseCudaEvaluator, SelfEnergyThreeControlsAndChangedMasks) {
  for (int partition : {0, 1, 2}) {
    SCOPED_TRACE(partition);
    auto ad                = atom(partition);
    auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
    auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
    std::vector<many_body_operator_complex> interactions{n("f", 0) + n("f", 1) + n("f", 2), c_dag("f", 0) * c("f", 1)};
    auto dyn                               = nda::zeros<dcomplex>(2, 2, 2);
    dyn(0, 0, 0)                           = .07;
    dyn(1, 1, 1)                           = {.03, .01};
    auto model                             = gpu::export_model(ad, coeff, interactions, dyn);
    auto quartet                           = direct_quartet(model);
    auto [rotated_quartet, rotated_labels] = bs::dynint::get_operators_and_interactions(ad, coeff, dyn, interactions);
    // Independent signed-map/CAR tests qualify the exact fields. Bound the old
    // eigenbasis round-trip noise separately from traversal and device error.
    for (std::size_t q = 0; q < quartet.Fs.size(); ++q)
      for (bool dag : {false, true}) {
        auto const &a = dag ? quartet.F_dags[q] : quartet.Fs[q];
        auto const &b = dag ? rotated_quartet.F_dags[q] : rotated_quartet.Fs[q];
        for (int column = 0; column < a.get_num_block_cols(); ++column)
          if (a.get_block_index(column) != -1) EXPECT_LE(nda::max_element(nda::abs(a.get_block(column) - b.get_block(column))), 4e-14);
      }
    nda::vector<double> poles{1.3, -.8};
    auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
    bs::DiagramEvaluator rotated(poles, coeff, G[0].mesh(), ad, interactions, dyn);
    bs::DiagramEvaluator exact(2., 40., 1e-10, poles, model.coefficients(), quartet, interactions.size());
    bs::DiagramEvaluator rebuilt(2., 40., 1e-10, poles, model.coefficients(), rotated_quartet, interactions.size());
    gpu::DiagramEvaluator device(model, poles, G[0].mesh());
    for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}},
                          nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}}, nda::array<int, 2>{{0, 4}, {1, 5}, {2, 6}, {3, 7}},
                          nda::array<int, 2>{{0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}}}) {
      long total = device.get_num_self_energy_backbones(topology);
      EXPECT_EQ(total, rotated.get_num_self_energy_backbones(topology));
      for (long first : {0L, total / 2, std::max(0L, total - 4)}) {
        SCOPED_TRACE(::testing::Message() << "order " << topology.extent(0) << " first " << first);
        long last   = topology.extent(0) == 1 ? total : std::min(total, first + (topology.extent(0) >= 4 ? 2 : 4));
        auto gold   = rotated.compute_self_energy_range(G, topology, first, last);
        auto direct = exact.compute_self_energy_range(G, topology, first, last);
        auto actual = device.compute_self_energy_range(G, topology, first, last);
        {
          SCOPED_TRACE("exact CPU versus rotated CPU");
          // E1's existing strict control covers orders 1--3. Newly sampled
          // higher orders characterize amplification of old round-trip noise.
          compare(direct, gold, topology.extent(0) <= 3 ? 1e-13 : 1e-12);
        }
        if (topology.extent(0) >= 4) {
          auto replay = rebuilt.compute_self_energy_range(G, topology, first, last);
          compare(replay, gold, 0.); // Same arrays reproduce the old result exactly.
        }
        {
          SCOPED_TRACE("CUDA versus exact CPU");
          compare(actual, direct);
        }
        if (topology.extent(0) == 1 && first == 0) EXPECT_GT(compare(actual, gold), 1e-20);
        auto count = device.last_path_counts();
        EXPECT_EQ(count.proposed, count.accepted + count.pruned);
        EXPECT_LE(device.last_contributing_paths(), count.accepted);
        // cudaMemGetInfo includes other CTest processes; only an explicitly
        // isolated run can compare this sampled physical delta with our bound.
        if (auto *isolated = std::getenv("XCA_BS_CHECK_MEMORY_PEAK"); isolated && std::string(isolated) == "1")
          EXPECT_LE(device.last_device_peak_delta(), device.self_energy_storage_bound() + 64ULL * 1024 * 1024);
      }
    }
    auto changed      = G;
    changed[1].data() = 0.;
    nda::array<int, 2> nca{{0, 1}};
    compare(device.compute_self_energy(changed, nca), rotated.compute_self_energy(changed, nca));
    compare(device.compute_self_energy(G, nca), rotated.compute_self_energy(G, nca));
    compare(device.compute_self_energy(G, nca, 1), rotated.compute_self_energy(G, nca, 1));
    EXPECT_THROW(device.compute_self_energy(G, nca, 2), std::out_of_range);
    auto bad               = G;
    bad[0].data()(0, 0, 0) = {std::numeric_limits<double>::quiet_NaN(), 0};
    EXPECT_THROW(device.compute_self_energy(bad, nca), std::invalid_argument);
    auto empty = device.compute_self_energy_range(G, nca, 2, 2);
    for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
    EXPECT_EQ(device.last_path_counts().proposed, 0);
    EXPECT_THROW(device.compute_self_energy_range(G, nca, -1, 0), std::out_of_range);
    gpu::DiagramEvaluator limited(model, poles, G[0].mesh(), 65536);
    EXPECT_THROW(limited.compute_self_energy(G, nca), std::length_error);
    compare(device.compute_self_energy(G, nca), rotated.compute_self_energy(G, nca));
  }
}
TEST(BlockSparseCudaEvaluator, MagnitudePolicyAcrossChunkBoundary) {
  gpu::DeviceOps ops;
  auto values     = nda::zeros<dcomplex>(2, 1, 4097);
  values(0, 0, 0) = 1e-16;
  gpu::Tensor input(values);
  EXPECT_FALSE(ops.any_abs_greater(input, 1e-16));
  values(1, 0, 4096) = {.8e-16, .8e-16};
  input              = values;
  EXPECT_TRUE(ops.any_abs_greater(input, 1e-16));
  EXPECT_THROW(ops.any_abs_greater(input, -1.), std::invalid_argument);
}
TEST(BlockSparseCudaEvaluator, OrdinaryManualNcaAndRangeAdditivity) {
  auto ad                = atom(1);
  auto [initial, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  auto model             = gpu::export_model(ad, coeff);
  auto quartet           = direct_quartet(model);
  nda::vector<double> poles{1.3, -.8};
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  gpu::DiagramEvaluator device(model, poles, G[0].mesh());
  triqs_xca::hyb::Hybridization hyb(G[0].mesh(), poles, coeff);
  bs::BlockDiagOpFun native(G);
  auto manual    = bs::manual::sigma_nca(hyb.values, hyb.values_reflect, native, quartet);
  auto manual_gf = triqs_xca::test_utils::BDOF_to_block_gf(manual, 2., 40., 1e-10);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  EXPECT_GT(compare(device.compute_self_energy(G, nca), manual_gf), 1e-20);
  auto total  = device.get_num_self_energy_backbones(oca);
  auto full   = device.compute_self_energy(G, oca);
  auto first  = device.compute_self_energy_range(G, oca, 0, total / 2);
  auto second = device.compute_self_energy_range(G, oca, total / 2, total);
  for (long b = 0; b < full.size(); ++b) first[b].data() += second[b].data();
  compare(first, full);
  EXPECT_THROW(gpu::DiagramEvaluator(model, nda::vector<double>{1.}, G[0].mesh()), std::invalid_argument);
  EXPECT_THROW(gpu::DiagramEvaluator(model, poles, G[0].mesh(), 0), std::length_error);
  EXPECT_THROW(gpu::DiagramEvaluator(model, poles, G[0].mesh(), 1024 * 1024 * 1024, 0), std::invalid_argument);
  gpu::DiagramEvaluator tuned(model, poles, G[0].mesh(), 1024 * 1024 * 1024, 64 * 1024 * 1024);
  EXPECT_EQ(tuned.self_energy_storage_bound() - device.self_energy_storage_bound(), 32ULL * 1024 * 1024);
  compare(tuned.compute_self_energy(G, nca), manual_gf);
  auto wrong_mesh = triqs_xca::test_utils::ad_to_atom_prop(ad, 3., 40., 1e-10);
  EXPECT_THROW(device.compute_self_energy(wrong_mesh, nca), std::invalid_argument);
  for (long b = 0; b < G.size(); ++b) G[b].data() = 0.;
  auto zero = device.compute_self_energy(G, nca);
  EXPECT_EQ(device.last_path_counts().accepted, 0);
  EXPECT_EQ(device.last_contributing_paths(), 0);
  for (long b = 0; b < zero.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(zero[b].data())), 0.);
}
#else
TEST(BlockSparseCudaEvaluator, DISABLED_RequiresCuda) {}
#endif
