// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fixture.hpp"
#include <fstream>
#include <string_view>

namespace block_sparse_bench {
  // Reference schedule arithmetic model, not measured traffic or elapsed work.
  // Complete structural candidates are recorded before the numerical cutoff.
  class WorkInventory {
    std::ofstream stream;
    std::size_t rows = 0;
    long rank;
    using Path = triqs_xca::block_sparse::SelfEnergyPath;
    static std::size_t mul(std::size_t a, std::size_t b) { return gpu::checked_multiply(a, b); }
    static std::size_t add(std::size_t a, std::size_t b) { return gpu::checked_add(a, b); }
    void row(Path const &p, std::size_t width, char const *kind, long stage, long M, long N, long K, long batches, std::size_t nnz, std::size_t flops,
             std::size_t dense, std::size_t bytes, int field = -1, int source = -1, int mix = -1) {
      if (rows == 200000) throw std::length_error("Work inventory exceeds 200000 rows; retained file is incomplete");
      stream << rows++ << ',' << p.flat_index << ',' << p.output_block << ',' << p.zero_set << ',' << width << ',' << kind << ',' << stage << ','
             << field << ',' << source << ',' << mix << ',' << M << ',' << N << ',' << K << ',' << batches << ',' << rank << ',' << nnz << ','
             << flops << ',' << dense << ',' << bytes << '\n';
      if (!stream) throw std::runtime_error("Work inventory write failed");
    }
    void dense(Path const &p, std::size_t width, char const *kind, long stage, long M, long N, long K, long batches) {
      auto flops = mul(8, mul(batches, mul(M, mul(N, K))));
      auto bytes = mul(16, mul(batches, add(mul(M, K), add(mul(K, N), mul(M, N)))));
      row(p, width, kind, stage, M, N, K, batches, 0, flops, flops, bytes);
    }
    void sparse(Path const &p, std::size_t width, char const *kind, long stage, gpu::SparseBlock const &op, long anchor, bool right, int field,
                int source, int mix = -1) {
      long M = right ? anchor : op.rows, N = right ? op.columns : anchor, K = right ? op.rows : op.columns;
      auto nnz   = op.values.size();
      auto csr   = add(mul((right ? op.columns : op.rows) + 1, sizeof(long)), mul(nnz, sizeof(long) + sizeof(nda::dcomplex)));
      auto bytes = add(csr, mul(16, mul(rank, add(right ? mul(M, K) : mul(K, N), mul(M, N)))));
      row(p, width, kind, stage, M, N, K, rank, nnz, mul(8, mul(rank, mul(nnz, anchor))), mul(8, mul(rank, mul(M, mul(N, K)))), bytes, field, source,
          mix);
    }
    void point(Path const &p, std::size_t width, char const *kind, long stage, long M, long N, long factor, long operands) {
      auto elements = mul(rank, mul(M, N));
      auto bytes    = mul(16, mul(operands, elements));
      if (std::string_view(kind).ends_with("weight")) bytes = add(bytes, mul(16, rank));
      row(p, width, kind, stage, M, N, 1, rank, 0, mul(factor, elements), mul(factor, elements), bytes);
    }

    public:
    WorkInventory(std::string const &file, long r) : rank(r) {
      if (file.empty()) return;
      stream.open(file);
      if (!stream) throw std::runtime_error("Cannot open work inventory");
      stream
         << "sequence,index,output_block,symmetry_set,set_width,kind,stage,field,source_block,mix_field,M,N,K,batches,rank,nnz,flops_model,dense_equivalent_flops,operand_footprint_bytes\n";
    }
    void record(gpu::OperatorModel const &model, Backbone &backbone, Path const &p) {
      if (!stream.is_open()) return;
      auto const &set = model.sets()[p.zero_set];
      auto width      = set.size();
      long s          = p.special_vertex;
      for (int v = 1; v < 2 * backbone.m; ++v) {
        if (v == s) continue;
        int f = backbone.get_vertex_orb(v), pole = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
        bool dag = backbone.has_vertex_dag(v), bar = backbone.has_vertex_bar(v);
        auto const &op = bar ? model.barred(f, pole, dag) : model.field(f, dag);
        sparse(p, width, bar ? "vertex_bar" : "vertex_field", v, op.blocks[p.propagator_blocks[v - 1]], p.dimensions[v < s ? 1 : 0], false, f,
               p.propagator_blocks[v - 1]);
        if (backbone.get_vertex_Ksign(v) != 0) point(p, width, "vertex_weight", v, p.dimensions[v + 1], p.dimensions[v < s ? 1 : 0], 6, 2);
      }
      for (int e = 1; e < 2 * backbone.m - 1; ++e) {
        long M = p.dimensions[e + 1], N = p.dimensions[e < s ? 1 : 0];
        for (int x = 0; x < backbone.m - 1; ++x)
          if (backbone.get_edge(e, x) != 0) point(p, width, "edge_weight", e, M, M, 6, 2);
        // Leading complex LU solve cost (pivoting/division/packing excluded).
        dense(p, width, "solve_leading", e, rank, M * M, rank, 1);
        dense(p, width, "solve_leading", e, rank, M * N, rank, 1);
        dense(p, width, "rank_H", e, rank, M * M, rank, 1);
        dense(p, width, "rank_H", e, rank, M * N, rank, 1);
        for (int product = 0; product < 3; ++product) dense(p, width, "spatial_convolution", e, M, N, M, rank);
        dense(p, width, "rank_Q", e, rank, M * N, rank, 1);
        dense(p, width, "rank_C", e, rank, M * N, rank, 1);
        point(p, width, "convolution_adds", e, M, N, 4, 6);
        point(p, width, "convolution_scale", e, M, N, 6, 2);
      }
      bool forward = !backbone.has_vertex_dag(0);
      int source   = p.propagator_blocks[s - 1];
      for (int i : set) {
        for (int j : set) {
          sparse(p, width, "special_right", s, model.field(j, !forward).blocks[p.output_block], p.dimensions[s], true, j, p.output_block, i);
          point(p, width, "special_weight", s, p.dimensions[s], p.dimensions[0], 6, 2);
          point(p, width, "special_mix", s, p.dimensions[s], p.dimensions[0], 2, 3);
        }
        sparse(p, width, "special_left", s, model.field(i, forward).blocks[source], p.dimensions[0], false, i, source);
        point(p, width, "special_accumulate", s, p.dimensions[s + 1], p.dimensions[0], 2, 3);
      }
      point(p, width, "final_scales", 2 * backbone.m, p.dimensions[0], p.dimensions[0], 12, 4);
      // Eligibility is numerical and cannot be inferred by this all-live visitor.
      point(p, width, "possible_output_sum", 2 * backbone.m, p.dimensions[0], p.dimensions[0], 2, 3);
    }
  };
} // namespace block_sparse_bench
