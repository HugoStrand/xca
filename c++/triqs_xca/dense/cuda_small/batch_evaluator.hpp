// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dlr_ops.hpp"
#ifdef XCA_HAVE_CUDA_SMALL
#include <functional>
#include <triqs/gfs.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>
namespace triqs_xca::dense::cuda_small {
  class BatchEvaluator {
    using gf_t     = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
    using gf_vt    = triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>;
    using Backbone = triqs_xca::backbone::Backbone;
    triqs::mesh::dlr_imtime mesh_;
    double beta_;
    triqs_xca::hyb::Hybridization hyb_;
    int r_, N_, n_, B_;
    DlrOps dlr_;
    tensor G_, T_, U_, tmp_, GKt_, gc_, tc_, Tmu_, parity_tmp_, lane_;
    tensor_view sum_;
    std::size_t budget_;
    nda::cuarray<complex, 4> Tkaps_, bars_, bars_refl_;
    tensor Fs_, Fd_;
    std::vector<std::unique_ptr<nda::cuvector<complex>>> weights_;
    std::vector<std::unique_ptr<DeviceOps::LaneTables>> tables_;
    std::vector<std::function<void()>> schedule_;
    std::vector<Backbone> backbones_;
    int active_ = 0, weight_cursor_ = 0, table_cursor_ = 0;
    std::size_t chunks_ = 0, cache_builds_ = 0;
    bool cache_enabled_ = true;
    bool cached_pairs_ = false;
    std::vector<int> cached_topology_;
    std::vector<long> cached_indices_;
    static int capacity(int b);
    public:
    BatchEvaluator(triqs::mesh::dlr_imtime mesh, cppdlr::imtime_ops const &itops, nda::vector_const_view<double> poles,
                     nda::array_const_view<complex, 3> coefficients, triqs_xca::dense::FSet fset, int b, std::size_t budget);
    matrix_const_view op(Backbone &b, int v, int o, int p = -1) const;
    nda::cuvector<complex> *add_weight(tensor_const_view in, tensor_view out, std::function<complex(int, int)> const &value, bool enqueue = true);
    void add_vertex(tensor_const_view in, tensor_view out, bool left, std::function<matrix_const_view(int)> const &value, complex beta = 0.);
    void vertex(int v);
    void edge(int e);
    void special(int v, tensor_view target);
    double prefactor(Backbone &b) const;
    void prepare(nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, bool pairs);

    ~BatchEvaluator() { try { dlr_.finish(); } catch (...) {} }
    void accumulate(tensor_const_view g, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> indices, tensor_view output, bool pairs);
    std::size_t workspace_bytes() const;
    void set_budget(std::size_t bytes) { budget_ = bytes; }
    int batch_capacity() const { return B_; }
    void reserve_metadata(std::size_t additional) const;
  };
} // namespace triqs_xca::dense::cuda_small
#endif
