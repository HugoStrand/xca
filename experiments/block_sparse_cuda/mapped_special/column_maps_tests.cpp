// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "column_maps.hpp"
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
