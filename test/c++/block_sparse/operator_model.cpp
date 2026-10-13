#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/operator_model.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include "test_utils/block_sparse.hpp"

namespace {
  namespace bs  = triqs_xca::block_sparse;
  namespace gpu = bs::cuda;
  using nda::dcomplex;
  constexpr auto all = nda::range::all;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  void compare_model(gpu::OperatorModel const &model, bs::BlockOpSymQuartet const &gold) {
    EXPECT_LE(model.host_storage_bytes(), model.host_storage_bound_bytes());
    EXPECT_LT(model.sparse_payload_bytes(), model.host_storage_bytes());
    ASSERT_EQ(model.sets().size(), gold.Fs.size());
    for (int i = 0; i < model.num_fields(); ++i) {
      ASSERT_EQ(model.labels()[i], gold.sym_set_labels(i));
      ASSERT_EQ(model.within_set()[i], gold.sym_set_inds(i));
      int group = model.labels()[i], within = model.within_set()[i];
      for (bool dag : {false, true}) {
        auto const &field = model.field(i, dag);
        auto const &op    = dag ? gold.F_dags[group] : gold.Fs[group];
        for (int b = 0; b < op.get_num_block_cols(); ++b) {
          ASSERT_EQ(field.connection[b], op.get_block_index(b));
          if (field.connection[b] == -1) continue;
          auto actual = field.blocks[b].to_dense();
          EXPECT_LE(nda::max_element(nda::abs(actual - op.get_block(b)(within, all, all))), 1e-14);
        }
        for (long p = 0; p < model.num_poles(); ++p) {
          auto const &bar      = model.barred(i, p, dag);
          auto const &expected = dag ? gold.F_dag_bars[group] : gold.F_bars_refl[group];
          for (int b = 0; b < expected.get_num_block_cols(); ++b) {
            ASSERT_EQ(bar.connection[b], expected.get_block_index(b));
            if (bar.connection[b] == -1) continue;
            auto actual = bar.blocks[b].to_dense();
            EXPECT_LE(nda::max_element(nda::abs(actual - expected.get_block(b)(within, p, all, all))), 1e-14);
          }
        }
      }
    }
  }
} // namespace
TEST(SparseModel, ComplexWeightedBarsAndOwnedCoefficients) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("f", 1);
  fops.insert("f", 0);
  many_body_operator_complex Q = n("f", 0) + n("f", 1);
  many_body_operator_complex H =
     .2 * n("f", 0) + .7 * n("f", 1) + dcomplex{.1, .2} * c_dag("f", 0) * c("f", 1) + dcomplex{.1, -.2} * c_dag("f", 1) * c("f", 0);
  triqs::atom_diag::atom_diag<true> ad(H, fops, std::vector<many_body_operator_complex>{Q});
  nda::array<dcomplex, 3> coeff(2, 2, 2);
  for (long p = 0; p < 2; ++p)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 2; ++j) coeff(p, i, j) = {.3 + .1 * p + .03 * i + .07 * j, .04 * (i - j) + .02 * p};
  auto model          = gpu::export_model(ad, coeff);
  auto [gold, labels] = bs::atom_diag::get_operators(ad, coeff);
  compare_model(model, gold);
  EXPECT_EQ(model.num_interactions(), 0);
  auto retained  = model.coefficients()(0, 0, 1);
  coeff(0, 0, 1) = 999.;
  EXPECT_EQ(model.coefficients()(0, 0, 1), retained);
  EXPECT_THROW(model.field(-1), std::out_of_range);
  EXPECT_THROW(model.barred(0, 2, true), std::out_of_range);
  EXPECT_THROW(gpu::export_model(ad, coeff, 1), std::length_error);
}
TEST(SparseModel, MixedBlocksAndDynamicOperators) {
  auto ad                = triqs_xca::test_utils::unequal_sym_set_model(true);
  int n_hyb              = ad.get_fops().size();
  auto [quartet, labels] = bs::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, n_hyb, n_hyb));
  auto coeff             = triqs_xca::test_utils::sym_set_diagonal_hyb(labels, 2);
  std::vector<many_body_operator_complex> ops{n("A", 0) + n("A", 1), c_dag("A", 0) * c("A", 1)};
  nda::array<dcomplex, 3> dyn  = nda::zeros<dcomplex>(2, 2, 2);
  dyn(0, 0, 0)                 = .2;
  dyn(1, 0, 0)                 = .3;
  dyn(0, 1, 1)                 = {.1, .04};
  dyn(1, 1, 1)                 = .15;
  auto model                   = gpu::export_model(ad, coeff, ops, dyn);
  auto [gold, combined_labels] = bs::dynint::get_operators_and_interactions(ad, coeff, dyn, ops);
  compare_model(model, gold);
  EXPECT_EQ(model.num_interactions(), 2);
  EXPECT_EQ(model.num_fields(), 5);
  EXPECT_THROW(gpu::export_model(ad, nda::zeros<dcomplex>(2, 2, 2)), std::invalid_argument);
  EXPECT_THROW(gpu::export_model(ad, coeff, ops, nda::zeros<dcomplex>(1, 2, 2)), std::invalid_argument);
}
TEST(SparseModel, ScalarSetsZeroCoefficientsAndCrossSetPolicy) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("f", 0);
  fops.insert("f", 1);
  triqs::atom_diag::atom_diag<false> ad(.2 * n("f", 0) + .7 * n("f", 1), fops);
  nda::array<dcomplex, 3> coeff = nda::zeros<dcomplex>(2, 2, 2);
  auto model                    = gpu::export_model(ad, coeff);
  auto [gold, labels]           = bs::atom_diag::get_operators(ad, coeff);
  compare_model(model, gold);
  for (int i = 0; i < 2; ++i)
    for (bool dag : {false, true}) EXPECT_EQ(model.barred(i, 0, dag).nonzeros(), 0);
  coeff(0, 0, 0) = 1.;
  coeff(0, 1, 1) = 1.;
  coeff(0, 0, 1) = .1;
  EXPECT_THROW(gpu::export_model(ad, coeff), std::invalid_argument);
  coeff(0, 0, 1)     = .5 * bs::sym_set_coupling_tol;
  auto remainder     = gpu::export_model(ad, coeff);
  auto [accepted, l] = bs::atom_diag::get_operators(ad, coeff);
  compare_model(remainder, accepted);
  coeff(0, 0, 0) = {std::numeric_limits<double>::quiet_NaN(), 0.};
  EXPECT_THROW(gpu::export_model(ad, coeff), std::invalid_argument);
}
TEST(SparseMemory, CheckedLargeSectorInventory) {
  std::vector<long> dims{1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1};
  auto memory = gpu::inventory_memory(26, dims, 14);
  EXPECT_EQ(memory.function, 16688505600ULL);
  EXPECT_EQ(memory.maximum_rank_buffer, 4899907584ULL);
  EXPECT_EQ(memory.legacy_scratch, 88198336512ULL);
  EXPECT_EQ(memory.legacy_correlator_sides, 146997227520ULL);
  EXPECT_THROW(gpu::inventory_memory(-1, dims, 14), std::invalid_argument);
  EXPECT_THROW(gpu::tensor_bytes(26, std::numeric_limits<long>::max(), 3432), std::overflow_error);
  EXPECT_THROW(gpu::checked_add(std::numeric_limits<std::size_t>::max(), 1), std::overflow_error);
  EXPECT_THROW(gpu::checked_multiply(std::numeric_limits<std::size_t>::max(), 2), std::overflow_error);
  auto scalar = gpu::inventory_memory(26, std::vector<long>(16384, 1), 1);
  EXPECT_EQ(scalar.function, 6815744ULL);
}
