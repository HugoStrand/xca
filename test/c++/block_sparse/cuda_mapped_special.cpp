// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/operator_column_maps.hpp>
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
#include <triqs_xca/block_sparse/cuda/kernels/kernels.hpp>
#endif
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  gpu::DiagramEvaluator make_mapped(gpu::OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh,
                                    std::size_t budget = 1024ULL * 1024 * 1024, std::size_t sparse = 32ULL * 1024 * 1024,
                                    gpu::SparseBackend backend = gpu::SparseBackend::Library) {
    return gpu::DiagramEvaluator(std::move(model), poles, mesh, budget, sparse, backend, gpu::SpecialBackend::Mapped);
  }
  void compare(block_sparse_bench::Gf const &a, block_sparse_bench::Gf const &b) {
    ASSERT_EQ(a.size(), b.size());
    for (long block = 0; block < a.size(); ++block)
      EXPECT_LE(nda::max_element(nda::abs(a[block].data() - b[block].data())), 1e-13 * nda::max_element(nda::abs(b[block].data())) + 1e-30);
  }
} // namespace
TEST(BlockSparseCudaMapped, MultiFieldRectangularPathsChangedInputAndMasks) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  auto model = gpu::export_model(ad, coeff);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &s) { return s.size() == 4; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  for (auto backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered}) {
    if (backend == gpu::SparseBackend::Ordered && !gpu::ordered_sparse_available()) continue;
    gpu::DiagramEvaluator control(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    auto candidate = make_mapped(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    for (int update = 0; update < 3; ++update) {
      if (update == 1)
        for (long b = 0; b < G.size(); ++b) G[b].data() *= nda::dcomplex{.7, .2};
      if (update == 2) G[2].data() = 0.;
      compare(candidate.compute_self_energy_range(G, topology, 0, 4), control.compute_self_energy_range(G, topology, 0, 4));
      EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
      EXPECT_EQ(candidate.last_path_counts().accepted, control.last_path_counts().accepted);
    }
  }
}
TEST(BlockSparseCudaMapped, BosonicOffDiagonalMixingPreservesReflectedOrder) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  auto ad    = block_sparse_bench::atom(2, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  using triqs::operators::n;
  std::vector<triqs::operators::many_body_operator_complex> operators{n("f", 0) + 1e-8, n("f", 1) + 1e-8};
  auto dyn     = nda::zeros<nda::dcomplex>(2, 2, 2);
  dyn(0, 0, 0) = .03;
  dyn(1, 1, 1) = .02;
  dyn(0, 0, 1) = {.007, .004};
  dyn(0, 1, 0) = std::conj(dyn(0, 0, 1));
  dyn(1, 0, 1) = {.003, -.006};
  dyn(1, 1, 0) = std::conj(dyn(1, 0, 1));
  auto model   = gpu::export_model(ad, coeff, operators, dyn);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &s) { return s.size() == 2 && s[0] >= 2; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  for (auto backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered}) {
    if (backend == gpu::SparseBackend::Ordered && !gpu::ordered_sparse_available()) continue;
    gpu::DiagramEvaluator control(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    auto candidate = make_mapped(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    for (int update = 0; update < 2; ++update) {
      if (update)
        for (long b = 0; b < G.size(); ++b) G[b].data() *= nda::dcomplex{.8, -.1};
      compare(candidate.compute_self_energy(G, topology), control.compute_self_energy(G, topology));
      EXPECT_EQ(candidate.last_path_counts().accepted, control.last_path_counts().accepted);
      EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
    }
  }
}
TEST(BlockSparseCudaMapped, MetadataAdmissionAndEmptyRange) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  auto ad    = block_sparse_bench::atom(4, "number");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator control(model, poles, mesh);
  auto candidate = make_mapped(model, poles, mesh);
  EXPECT_EQ(candidate.self_energy_storage_bound() - control.self_energy_storage_bound(),
            4 * (6 * (sizeof(long) + sizeof(nda::dcomplex)) + mesh.dlr_it().rank() * sizeof(nda::dcomplex)));
  auto limited = make_mapped(model, poles, mesh, candidate.self_energy_storage_bound() - 1);
  EXPECT_THROW(limited.compute_self_energy_range(G, topology, 0, 1), std::length_error);
  EXPECT_EQ(limited.last_path_counts().accepted, 0);
  EXPECT_EQ(limited.last_device_peak_delta(), 0);
  auto empty = limited.compute_self_energy_range(G, topology, 0, 0);
  for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
  compare(candidate.compute_self_energy(G, topology), control.compute_self_energy(G, topology));
}
TEST(BlockSparseCudaMapped, NonMappingBosonicSetUsesReference) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  using namespace triqs::operators;
  many_body_operator_complex Q;
  for (int i = 0; i < 4; ++i) Q += n("f", i);
  many_body_operator_complex H0 = c_dag("f", 0) * c("f", 1) + c_dag("f", 1) * c("f", 0);
  many_body_operator_complex H1 = c_dag("f", 2) * c("f", 3) + c_dag("f", 3) * c("f", 2);
  std::vector<many_body_operator_complex> operators{Q + 1e-8 + H0, Q + 1e-8 + H1};
  auto dyn      = nda::zeros<nda::dcomplex>(2, 2, 2);
  dyn(0, 0, 0)  = .02;
  dyn(1, 1, 1)  = .03;
  auto model    = gpu::export_model(ad, coeff, operators, dyn);
  bool rejected = false;
  for (auto const &set : model.sets())
    if (set.size() == 2 && set[0] >= 4)
      for (long b = 0; b < long(model.dimensions().size()); ++b) {
        std::vector<gpu::SparseBlock const *> blocks;
        for (int field : set) blocks.push_back(&model.field(field).blocks[b]);
        if (blocks[0]->rows > 0 && !gpu::make_operator_column_maps(blocks)) rejected = true;
      }
  ASSERT_TRUE(rejected);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator control(model, poles, mesh);
  auto candidate = make_mapped(model, poles, mesh);
  std::vector<bool> live(G.size());
  for (long block = 0; block < G.size(); ++block) live[block] = nda::max_element(nda::abs(G[block].data())) >= 1e-16;
  long fallback_paths = 0;
  gpu::for_each_self_energy_path(model, poles, topology, live, 0, candidate.get_num_self_energy_backbones(topology),
                                 [&](triqs_xca::backbone::Backbone &backbone, triqs_xca::block_sparse::SelfEnergyPath const &path) {
                                   auto const &set = model.sets()[path.zero_set];
                                   if (set.size() != 2 || set[0] < 4) return;
                                   std::vector<gpu::SparseBlock const *> selected;
                                   for (int field : set)
                                     selected.push_back(&model.field(field, backbone.has_vertex_dag(0)).blocks[path.output_block]);
                                   if (!gpu::make_operator_column_maps(selected)) ++fallback_paths;
                                 });
  ASSERT_GT(fallback_paths, 0);
  compare(candidate.compute_self_energy(G, topology), control.compute_self_energy(G, topology));
  EXPECT_EQ(candidate.last_path_counts().accepted, control.last_path_counts().accepted);
  EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
}
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
TEST(BlockSparseCudaMapped, KernelMatchesSparseWeightingWithComplexEmptyColumnsAndTails) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  using Complex = nda::dcomplex;
  auto owner    = std::make_shared<gpu::DeviceOps>();
  long rank = 3, rows = 4, input_columns = 5, columns = 17, fields = 3;
  std::vector<gpu::SparseBlock> blocks(fields);
  for (long field = 0; field < fields; ++field) {
    auto &block   = blocks[field];
    block.rows    = input_columns;
    block.columns = columns;
    block.row_offsets.clear();
    block.row_offsets.push_back(0);
    for (long row = 0; row < input_columns; ++row) {
      for (long column = 0; column < columns; ++column)
        if (column % 7 != 0 && (column + field) % input_columns == row) {
          block.column_indices.push_back(column);
          block.values.push_back({.2 * (field + 1), -.03 * (column + 1)});
        }
      block.row_offsets.push_back(block.values.size());
    }
  }
  auto maps = gpu::make_operator_column_maps({&blocks[0], &blocks[1], &blocks[2]});
  ASSERT_TRUE(maps);
  nda::array<Complex, 3> host(rank, rows, input_columns);
  for (long t = 0; t < rank; ++t)
    for (long row = 0; row < rows; ++row)
      for (long column = 0; column < input_columns; ++column) host(t, row, column) = {.13 * (t + row + 1), -.02 * (column + 1)};
  gpu::Tensor input(host), reference(rank, rows, columns), actual(reference.shape()), kap(reference.shape()), weighted(reference.shape());
  nda::cuarray<long, 2> indices(maps->sources);
  nda::cuarray<Complex, 2> values(maps->values);
  for (int update = 0; update < 2; ++update) {
    nda::array<Complex, 2> weights(rank, fields);
    for (long t = 0; t < rank; ++t)
      for (long field = 0; field < fields; ++field) weights(t, field) = {.3 * (field + 1), .07 * (t + update + 1)};
    nda::cuarray<Complex, 2> device_weights(weights);
    owner->zero(reference);
    for (long field = 0; field < fields; ++field) {
      gpu::DeviceSparseBlock op(owner, blocks[field]);
      op.apply_right(input, kap);
      nda::cuvector<Complex> w(nda::vector<Complex>(weights(nda::range::all, field)));
      owner->weight_rows(kap, w, weighted);
      owner->axpy(1., weighted, reference);
    }
    owner->wait_for_default_uploads();
    ASSERT_EQ(gpu::native::mapped_weighted_mu(input.data(), indices.data(), values.data(), device_weights.data(), actual.data(), rank, rows,
                                              input_columns, columns, fields, owner->stream()),
              cudaSuccess);
    owner->finish();
    auto gold = nda::to_host(reference), result = nda::to_host(actual);
    EXPECT_LE(nda::max_element(nda::abs(result - gold)), 1e-13 * nda::max_element(nda::abs(gold)) + 1e-30);
  }
}
TEST(BlockSparseCudaMapped, CancellationKeepsFieldOrderAndEmptyColumns) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  using Complex = nda::dcomplex;
  auto owner    = std::make_shared<gpu::DeviceOps>();
  long rank = 3, rows = 2, columns = 19;
  nda::array<Complex, 3> host(rank, rows, 2);
  for (long t = 0; t < rank; ++t)
    for (long row = 0; row < rows; ++row) {
      host(t, row, 0) = {1e16, -1e16};
      host(t, row, 1) = {1., 3.};
    }
  nda::array<long, 2> indices(3, columns);
  nda::array<Complex, 2> values(3, columns), weights(rank, 3);
  values() = 1.;
  for (long field = 0; field < 3; ++field)
    for (long column = 0; column < columns; ++column) indices(field, column) = column == columns - 1 ? -1 : (field == 2 ? 1 : 0);
  for (long t = 0; t < rank; ++t) {
    weights(t, 0) = 1.;
    weights(t, 1) = -1.;
    weights(t, 2) = 1.;
  }
  gpu::Tensor input(host), output(rank, rows, columns);
  nda::cuarray<long, 2> device_indices(indices);
  nda::cuarray<Complex, 2> device_values(values), device_weights(weights);
  owner->wait_for_default_uploads();
  ASSERT_EQ(gpu::native::mapped_weighted_mu(input.data(), device_indices.data(), device_values.data(), device_weights.data(), output.data(), rank,
                                            rows, 2, columns, 3, owner->stream()),
            cudaSuccess);
  owner->finish();
  auto result = nda::to_host(output);
  for (long t = 0; t < rank; ++t)
    for (long row = 0; row < rows; ++row)
      for (long column = 0; column < columns; ++column) EXPECT_EQ(result(t, row, column), column == columns - 1 ? Complex{0.} : Complex(1., 3.));
}
#endif
TEST(BlockSparseCudaMapped, CompletedNcaPathCutoffNeighborhood) {
  if (!gpu::mapped_special_available()) GTEST_SKIP() << "Native mapped kernels unavailable";
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = nda::zeros<nda::dcomplex>(1, 4, 4);
  for (int field = 0; field < 4; ++field) coeff(0, field, field) = .01;
  auto model = gpu::export_model(ad, coeff);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 10., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{0.};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator control(model, poles, mesh);
  auto candidate = make_mapped(model, poles, mesh);
  auto fill      = [&](double amplitude) {
    for (long b = 0; b < G.size(); ++b) G[b].data() = 0.;
    for (long t = 0; t < mesh.dlr_it().rank(); ++t)
      for (long i = 0; i < G[2].data().extent(1); ++i) G[2].data()(t, i, i) = amplitude;
  };
  fill(1.);
  auto unit     = control.compute_self_energy(G, topology, 0);
  double factor = 0.;
  for (long b = 0; b < unit.size(); ++b) factor = std::max(factor, nda::max_element(nda::abs(unit[b].data())));
  ASSERT_GT(factor, 0.);
  double boundary = 1e-16 / factor;
  ASSERT_GT(boundary, 1e-16);
  double amplitude = boundary;
  for (int i = 0; i < 4; ++i) amplitude = std::nextafter(amplitude, 0.);
  bool kept = false, rejected = false;
  for (int i = 0; i < 9; ++i) {
    fill(amplitude);
    auto reference = control.compute_self_energy(G, topology, 0);
    auto actual    = candidate.compute_self_energy(G, topology, 0);
    compare(actual, reference);
    EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
    kept |= control.last_contributing_paths() > 0;
    rejected |= control.last_contributing_paths() == 0;
    amplitude = std::nextafter(amplitude, 1.);
  }
  EXPECT_TRUE(kept);
  EXPECT_TRUE(rejected);
}

TEST(BlockSparseCudaMapped, InvalidPolicyAndUnavailableCapability) {
  auto ad    = block_sparse_bench::atom(2, "occupation");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  nda::vector<double> poles{-1.5, .6};
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  EXPECT_THROW(gpu::DiagramEvaluator(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Library,
                                     static_cast<gpu::SpecialBackend>(-1)),
               std::invalid_argument);
  if (!gpu::mapped_special_available()) EXPECT_THROW(make_mapped(model, poles, mesh), std::runtime_error);
}
#else
#include <gtest/gtest.h>
TEST(BlockSparseCudaMapped, DISABLED_RequiresCuda) {}
#endif
