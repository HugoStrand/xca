// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "column_maps.hpp"
#include "fock_fields.hpp"
#include <gtest/gtest.h>
TEST(ColumnMaps, RectangularComplexEmptyColumnsAndUnsupportedStructure) {
  using triqs_xca::block_sparse::cuda::SparseBlock;
  SparseBlock block{3, 4, {0, 2, 2, 3}, {0, 2, 1}, {{1., .2}, {1e-25, -3e-26}, {-1., .4}}};
  auto maps = mapped_special_experiment::column_maps({&block});
  ASSERT_TRUE(maps);
  EXPECT_EQ(maps->sources(0, 0), 0);
  EXPECT_EQ(maps->sources(0, 1), 2);
  EXPECT_EQ(maps->sources(0, 2), 0);
  EXPECT_EQ(maps->sources(0, 3), -1);
  EXPECT_EQ(maps->values(0, 2), block.values[1]);
  SparseBlock unsupported{3, 4, {0, 1, 2, 2}, {0, 0}, {{1., 0.}, {2., 0.}}};
  EXPECT_FALSE(mapped_special_experiment::column_maps({&unsupported}));
  EXPECT_FALSE(mapped_special_experiment::column_maps({&block, &unsupported}));
  EXPECT_FALSE(mapped_special_experiment::column_maps({}));
}

TEST(ColumnMaps, EmptyPayloadAndDifferentShapes) {
  using triqs_xca::block_sparse::cuda::SparseBlock;
  SparseBlock empty{3, 4, {0, 0, 0, 0}, {}, {}};
  auto maps = mapped_special_experiment::column_maps({&empty});
  ASSERT_TRUE(maps);
  for (long column = 0; column < 4; ++column) {
    EXPECT_EQ(maps->sources(0, column), -1);
    EXPECT_EQ(maps->values(0, column), nda::dcomplex{0.});
  }
  SparseBlock different{3, 5, {0, 0, 0, 0}, {}, {}};
  EXPECT_FALSE(mapped_special_experiment::column_maps({&empty, &different}));
  EXPECT_FALSE(mapped_special_experiment::column_maps({nullptr}));
  EXPECT_FALSE(mapped_special_experiment::column_maps({&empty, nullptr}));
}
TEST(FockFields, KnownFourModeSignsAndFourteenModeCompleteness) {
  auto small = mapped_special_experiment::fock_fields(4, false);
  auto a = small[0].to_dense(), b = small[1].to_dense();
  EXPECT_EQ(a.shape(), (std::array<long, 2>{4, 6}));
  EXPECT_EQ(a(1, 0), 1.);
  EXPECT_EQ(a(2, 1), 1.);
  EXPECT_EQ(a(3, 3), 1.);
  EXPECT_EQ(b(0, 0), -1.);
  EXPECT_EQ(b(2, 2), 1.);
  EXPECT_EQ(b(3, 4), 1.);
  auto fields = mapped_special_experiment::fock_fields(14, false);
  ASSERT_EQ(fields.size(), 14);
  EXPECT_EQ(fields[0].rows, 3003);
  EXPECT_EQ(fields[0].columns, 3432);
  std::vector<long> column_counts(3432), row_counts(3003);
  for (auto const &field : fields) {
    EXPECT_EQ(field.values.size(), 1716);
    for (long row = 0; row < field.rows; ++row) {
      row_counts[row] += field.row_offsets[row + 1] - field.row_offsets[row];
      for (long k = field.row_offsets[row]; k < field.row_offsets[row + 1]; ++k) {
        ++column_counts[field.column_indices[k]];
        EXPECT_EQ(std::abs(field.values[k]), 1.);
      }
    }
  }
  for (long count : column_counts) EXPECT_EQ(count, 7);
  for (long count : row_counts) EXPECT_EQ(count, 8);
  auto creation = mapped_special_experiment::fock_fields(14, true);
  for (long field = 0; field < 14; ++field) {
    auto expected = fields[field].adjoint();
    EXPECT_EQ(creation[field].row_offsets, expected.row_offsets);
    EXPECT_EQ(creation[field].column_indices, expected.column_indices);
    EXPECT_EQ(creation[field].values, expected.values);
  }
}
