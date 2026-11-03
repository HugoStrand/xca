#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nda/nda.hpp>
#include <triqs/gfs.hpp>
#include <triqs_xca/block_sparse/action.hpp>
#include <triqs_xca/block_sparse/backbone_paths.hpp>
#include "test_utils/evaluators.hpp"
#include "kernels.hpp"

namespace block_sparse_layout {

  namespace fixture_detail {

    namespace bs = triqs_xca::block_sparse;
    namespace tu = triqs_xca::test_utils;
    using nda::range;
    inline constexpr auto all = range::all;

    inline tu::EvalModel model(std::string const &name) {
      if (name != "density4") return tu::build_eval_model(name);
      triqs::hilbert_space::fundamental_operator_set fops;
      triqs::operators::many_body_operator_complex H;
      for (int i = 0; i < 4; ++i) {
        fops.insert("a", i);
        H += (0.13 + 0.17 * i) * triqs::operators::n("a", i);
        for (int j = 0; j < i; ++j) H += 0.6 * triqs::operators::n("a", i) * triqs::operators::n("a", j);
      }
      return tu::make_eval_model(name, tu::ad_t(H, fops));
    }

    inline nda::matrix<dcplx> vertex_matrix(triqs_xca::Action<bs::BlockOpSymQuartet> const &action, bs::diagram::CorrelatorBackbonePath const &path,
                                            int vertex) {
      auto const &v   = path.backbone.vertices[vertex];
      auto const &ops = action.operators;
      int set = ops.sym_set_labels(v.flavour), index = ops.sym_set_inds(v.flavour);
      if (v.barred) {
        auto const &op = v.adjoint ? ops.F_dag_bars[set] : ops.F_bars_refl[set];
        return nda::matrix<dcplx>(op.get_block(path.sector(vertex))(index, v.pole_index, all, all));
      }
      auto const &op = v.adjoint ? ops.F_dags[set] : ops.Fs[set];
      return nda::matrix<dcplx>(op.get_block(path.sector(vertex))(index, all, all));
    }

    inline Segment capture(std::string name, bool backward, triqs_xca::Action<bs::BlockOpSymQuartet> const &action, bs::BlockDiagOpFun const &G,
                           bs::diagram::CorrelatorBackbonePath const &path) {
      auto weighted_edge = [&](int edge) {
        nda::array<dcplx, 3> result(G.get_block(path.edge_sector(edge)));
        triqs_xca::apply_kernel_product(result, path.backbone.edges[edge].kernels, action.kernel);
        return result;
      };
      auto const &interval = path.range;
      Segment result{std::move(name),
                     backward,
                     nda::max_element(action.block_dims),
                     weighted_edge(backward ? interval.edge_end - 1 : interval.first_edge),
                     {},
                     1.};
      auto append = [&](int vertex, int edge) {
        nda::vector<double> kernel;
        if (path.backbone.vertices[vertex].kernel_sign != 0) {
          auto values = nda::ones<dcplx>(action.tau_mesh.size(), 1, 1);
          triqs_xca::apply_kernel(values, path.backbone.vertices[vertex], action.kernel, backward);
          kernel.resize(action.tau_mesh.size());
          for (long t = 0; t < kernel.size(); ++t) kernel(t) = values(t, 0, 0).real();
        }
        result.steps.push_back({vertex_matrix(action, path, vertex), weighted_edge(edge), std::move(kernel)});
      };
      if (backward) {
        for (int v = interval.vertex_end - 1; v >= interval.first_vertex; --v) append(v, v - 1);
      } else {
        for (int v = interval.first_vertex; v < interval.vertex_end; ++v) append(v, v);
        result.prefactor = triqs_xca::tau_kernel_prefactor(path.backbone.kernel_prefactor, action.kernel) * path.backbone.direction_sign;
      }
      return result;
    }

    /// Prefer paths that exercise changing dimensions, with larger blocks breaking ties.
    inline long score(Segment const &segment) {
      long current = segment.initial.extent(1), changes = 0, largest = current;
      for (auto const &step : segment.steps) {
        long next = step.edge.extent(1);
        changes += next != current;
        largest = std::max(largest, next);
        current = next;
      }
      return 100 * changes + largest;
    }

    /// Smooth, complex, noncommuting matrices; all entries are deterministic.
    inline nda::array<dcplx, 3> synthetic_values(triqs::mesh::dlr_imtime const &mesh, long dimension, double seed) {
      nda::array<dcplx, 3> values(mesh.size(), dimension, dimension);
      auto tau = cppdlr::rel2abs(mesh.dlr_it().get_itnodes());
      for (long t = 0; t < mesh.size(); ++t)
        for (long i = 0; i < dimension; ++i)
          for (long j = 0; j < dimension; ++j) {
            dcplx off(std::sin(seed + 0.7 * i - 0.3 * j), std::cos(seed - 0.2 * i + 0.5 * j));
            values(t, i, j) = -std::exp(-0.4 * tau(t)) * ((i == j ? dcplx(1.) : dcplx(0.)) + 0.08 * off / dimension);
          }
      return values;
    }

    inline Segment synthetic_segment(triqs::mesh::dlr_imtime const &mesh, std::string name, bool backward, long capacity,
                                     std::vector<long> const &dimensions) {
      Segment result{std::move(name), backward, capacity, synthetic_values(mesh, dimensions.front(), 0.1), {}, backward ? 1. : -0.73};
      auto tau = cppdlr::rel2abs(mesh.dlr_it().get_itnodes());
      for (long s = 1; s < long(dimensions.size()); ++s) {
        long old = dimensions[s - 1], next = dimensions[s];
        nda::matrix<dcplx> vertex(backward ? old : next, backward ? next : old);
        for (long i = 0; i < vertex.extent(0); ++i)
          for (long j = 0; j < vertex.extent(1); ++j)
            vertex(i, j) = dcplx(std::sin(0.6 * i + 0.9 * j + s), std::cos(0.2 * i - 0.7 * j + 0.3 * s)) / std::sqrt(double(old));
        nda::vector<double> kernel(mesh.size());
        for (long t = 0; t < mesh.size(); ++t) kernel(t) = std::exp((backward ? -0.2 : 0.2) * tau(t));
        result.steps.push_back({std::move(vertex), synthetic_values(mesh, next, 0.4 + s), std::move(kernel)});
      }
      return result;
    }

  } // namespace fixture_detail

  /// Actual order-three correlator segments from the same models as evaluator tests.
  /// Preparation and path discovery are bounded and outside every timed region.
  inline std::vector<Segment> physical_segments(triqs::mesh::dlr_imtime const &mesh) {
    namespace bs      = triqs_xca::block_sparse;
    namespace tu      = triqs_xca::test_utils;
    namespace diagram = triqs_xca::diagram;
    std::vector<Segment> result;
    for (std::string const name : {"density4", "unequal_sets_nA0", "n_partition_spin_xyz"}) {
      auto model  = fixture_detail::model(name);
      auto action = bs::make_action(mesh, model.hyb_poles, model.hyb_coeffs, model.ad, model.dynint_ops, model.dynint_coeffs);

      // The test helper accepts a mutable itops even though it only reads it.
      auto itops  = mesh.dlr_it();
      auto atomic = tu::ad_to_atom_prop(model.ad, mesh.beta(), itops);
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks;
      for (int b = 0; b < model.ad.n_subspaces(); ++b) blocks.emplace_back(mesh, atomic.get_block(b));
      tu::bgf_t G0(blocks);
      auto dressed = tu::dressed_propagator(model.ad, mesh, G0);
      bs::BlockDiagOpFun G(tu::subspace_blocks(dressed, model.ad, mesh));
      bs::OperatorSupportTable support(action.operators);

      constexpr int order = 3;
      nda::array<int, 2> topology{{0, 3}, {1, 4}, {2, 5}};
      long count = diagram::count_diagrams(order, action.n_flavours(), action.poles.size(), diagram::Kind::correlator);
      std::mt19937_64 random{1789};
      std::vector<Segment> chosen(2);
      long scores[2] = {-1, -1};
      for (int sample = 0; sample < 128; ++sample) {
        long index    = std::uniform_int_distribution<long>{0, count - 1}(random);
        auto backbone = diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index);
        auto ranges   = backbone.endpoint_ranges();
        for (int direction = 0; direction < 2; ++direction) {
          auto interval = direction ? ranges.after_special : ranges.before_special;
          bs::visit_backbone_paths(G, backbone, action.operators, support, interval, [&](bs::diagram::CorrelatorBackbonePath const &path) {
            auto segment = fixture_detail::capture("physical_" + name + (direction ? "_right" : "_left"), direction, action, G, path);
            long value   = fixture_detail::score(segment);
            if (value > scores[direction]) {
              scores[direction] = value;
              chosen[direction] = std::move(segment);
            }
          });
        }
      }
      for (int direction = 0; direction < 2; ++direction) {
        if (scores[direction] < 0) throw std::runtime_error("No productive fixture for " + name);
        result.push_back(std::move(chosen[direction]));
      }
    }
    return result;
  }

  /// Explicit shape stress tests, not claims about a physical symmetry partition.
  inline std::vector<Segment> synthetic_segments(triqs::mesh::dlr_imtime const &mesh) {
    std::vector<Segment> result;
    for (bool backward : {false, true}) {
      std::string side = backward ? "_right" : "_left";
      result.push_back(fixture_detail::synthetic_segment(mesh, "synthetic_uniform1" + side, backward, 1, {1, 1, 1}));
      result.push_back(fixture_detail::synthetic_segment(mesh, "synthetic_uniform4" + side, backward, 4, {4, 4, 4}));
      result.push_back(fixture_detail::synthetic_segment(mesh, "synthetic_mixed_small_anchor1" + side, backward, 64, {1, 4, 16, 4, 1}));
      result.push_back(fixture_detail::synthetic_segment(mesh, "synthetic_mixed_small_anchor4" + side, backward, 64, {4, 1, 16, 4, 1}));
      result.push_back(fixture_detail::synthetic_segment(mesh, "synthetic_reaches64" + side, backward, 64, {4, 16, 64, 16, 4}));
      if (!backward) {
        auto full_scale                = fixture_detail::synthetic_segment(mesh, "synthetic_full_scale_anchor1" + side, false, 64, {1, 4, 16, 4, 1});
        full_scale.baseline_full_scale = true;
        result.push_back(std::move(full_scale));
      }
    }
    return result;
  }

} // namespace block_sparse_layout
