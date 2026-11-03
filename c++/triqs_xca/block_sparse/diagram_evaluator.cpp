#include "backbone_paths.hpp"
#include "self_energy_paths.hpp"
#include "correlator_paths.hpp"
#include "triqs_xca/action.hpp"
#include "triqs_xca/work_partition.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include <itertools/itertools.hpp>

#include "triqs_xca/block_sparse/atom_diag.hpp"

#include "triqs_xca/block_sparse/diagram_evaluator.hpp"
#include "triqs_xca/block_sparse/dynint.hpp"

#include "triqs_xca/operator_statistics.hpp"

namespace triqs_xca::block_sparse {

  using cppdlr::_;

  using nda::range;
  using nda::trace;
  using nda::linalg::matmul;

  namespace {

    // the leading index kap of Tkaps runs within one symmetry set, so the largest set is the bound rather than n
    long max_sym_set_size(BlockOpSymQuartet const &Fq) { return nda::max_element(Fq.sym_set_sizes); }

    template <triqs_xca::diagram::Kind K> void require_complete_path(diagram::BackbonePath<K> const &path) {
      path.range.validate();
      if (path.range != path.backbone.range()) throw std::invalid_argument("evaluation requires a complete backbone path");
      if (path.sectors.size() != path.range.vertex_end - path.range.first_vertex + 1 || path.sector_dimensions.size() != path.sectors.size())
        throw std::invalid_argument("backbone path sector storage does not match its range");
    }

    // A standalone completion has no segment cache. Copy only logical metadata;
    // numerical blocks remain in Action, G and the separately supplied observables.
    diagram::CorrelatorBackbonePath segment_path(diagram::CorrelatorBackbonePath const &path, triqs_xca::diagram::BackboneRange interval) {
      diagram::BackbonePath segment(path.backbone, interval);
      for (int slot = interval.first_vertex; slot <= interval.vertex_end; ++slot) {
        segment.sector(slot)    = path.sector(slot);
        segment.dimension(slot) = path.dimension(slot);
      }
      return segment;
    }

    void check_block_count(long n_blocks, long n_sub) {
      if (n_blocks != n_sub)
        throw std::invalid_argument("block_sparse::DiagramEvaluator: the propagator must have one block per invariant subspace, i.e. "
                                    + std::to_string(n_sub) + " blocks, got " + std::to_string(n_blocks));
    }

    // a propagator block has shape (r, d, d); only its rank r is checked where the subspace dimension d is unknown (d < 0)
    void check_block_shape(long b, std::array<long, 3> const &shape, long r, long dim) {
      if (shape[0] != r || (dim >= 0 && (shape[1] != dim || shape[2] != dim)))
        throw std::invalid_argument("block_sparse::DiagramEvaluator: block " + std::to_string(b) + " of the propagator has shape ("
                                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "), expected "
                                    + (dim >= 0 ?
                                          "(r, d, d) = (" + std::to_string(r) + ", " + std::to_string(dim) + ", " + std::to_string(dim) + ")" :
                                          "rank r = " + std::to_string(r)));
    }

    BlockDiagOpFun native_propagator(block_gf_view<dlr_imtime> G_ppsc, Action<BlockOpSymQuartet> const &action) {
      check_block_count(G_ppsc.size(), action.block_dims.size());
      for (long b = 0; b < action.block_dims.size(); ++b)
        check_block_shape(b, G_ppsc[b].data().shape(), action.tau_mesh.size(), action.block_dims(b));
      return BlockDiagOpFun(G_ppsc);
    }

    void check_propagator(BlockDiagOpFun const &Gt, Action<BlockOpSymQuartet> const &action) {
      check_block_count(Gt.get_num_block_cols(), action.block_dims.size());
      for (long b = 0; b < action.block_dims.size(); ++b) check_block_shape(b, Gt.get_block(b).shape(), action.tau_mesh.size(), action.block_dims(b));
    }

    block_gf<dlr_imtime> self_energy_result(BlockDiagOpFun const &Sigma, triqs::mesh::dlr_imtime const &mesh) {
      std::vector<triqs::gfs::gf<dlr_imtime>> blocks;
      for (int sector = 0; sector < Sigma.get_num_block_cols(); ++sector) blocks.emplace_back(mesh, Sigma.get_block(sector));
      return {blocks};
    }

    std::vector<BlockOp> single_particle_operators(bool adjoint, Action<BlockOpSymQuartet> const &action) {
      std::vector<BlockOp> result;
      // Flavour order is the external single-particle index, independently of symmetry sets.
      for (int o_ix = 0; o_ix < action.n_hyb; ++o_ix) {
        auto const &F =
           adjoint ? action.operators.F_dags[action.operators.sym_set_labels(o_ix)] : action.operators.Fs[action.operators.sym_set_labels(o_ix)];
        int i = static_cast<int>(action.operators.sym_set_inds(o_ix));
        std::vector<nda::array<dcomplex, 2>> blocks;
        for (int j = 0; j < F.get_num_block_cols(); ++j) {
          if (F.get_block_index(j) != -1) {
            blocks.emplace_back(F.get_block(j)(i, _, _));
          } else {
            blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
          }
        }
        nda::vector<int> block_indices = F.get_block_indices()(_);
        result.emplace_back(block_indices, blocks);
      }
      return result;
    }

    template <bool isComplex>
    std::vector<BlockOp> operator_matrices(std::vector<many_body_operator_complex> const &ops, triqs::atom_diag::atom_diag<isComplex> const &ad) {

      std::vector<BlockOp> result;

      for (auto const &op : ops) {
        auto op_blocks = triqs_xca::atom_diag::get_op_mat(ad, op);

        std::vector<nda::array<dcomplex, 2>> blocks;

        for (auto [b, op_block_mat] : itertools::enumerate(op_blocks.block_mat)) {

          int bc = op_blocks.connection(b);

          if (bc != -1) {
            auto UR                  = ad.get_unitary_matrix(b);
            auto UL                  = ad.get_unitary_matrix(bc);
            auto op_block_mat_transf = UL * op_block_mat * nda::conj(nda::transpose(UR));
            blocks.emplace_back(op_block_mat_transf);
          } else {
            blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
          }
        }

        nda::vector<int> block_indices = op_blocks.connection(_);

        result.emplace_back(block_indices, blocks);
      }
      return result;
    }

    nda::array_const_view<dcomplex, 2> endpoint_matrix(int flavour, bool adjoint, int source_sector, Action<BlockOpSymQuartet> const &action) {
      auto const &operators = action.operators;
      int set = operators.sym_set_labels(flavour), index = operators.sym_set_inds(flavour);
      auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
      return op.get_block(source_sector)(index, _, _);
    }

    template <triqs_xca::diagram::Kind K>
    nda::array_const_view<dcomplex, 2> vertex_matrix(diagram::BackbonePath<K> const &path, int vertex, Action<BlockOpSymQuartet> const &action) {
      auto const &v         = path.backbone.vertices[vertex];
      auto const &operators = action.operators;
      int set = operators.sym_set_labels(v.flavour), index = operators.sym_set_inds(v.flavour);
      if (v.barred) {
        auto const &op = v.adjoint ? operators.F_dag_bars[set] : operators.F_bars_refl[set];
        return op.get_block(path.sector(vertex))(index, v.pole_index, _, _);
      }
      return endpoint_matrix(v.flavour, v.adjoint, path.sector(vertex), action);
    }

    nda::array_const_view<dcomplex, 2> endpoint_coefficients(int late_flavour, bool reflected, Action<BlockOpSymQuartet> const &action) {
      if (reflected) return action.tau_values_reflect(_, _, late_flavour);
      return action.tau_values(_, late_flavour, _);
    }

    template <triqs_xca::diagram::Kind K>
    void multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, diagram::BackbonePath<K> const &path, int vertex, int columns,
                              Action<BlockOpSymQuartet> const &action) {
      auto const r = action.tau_mesh.size();
      auto before  = T_buf(_, range(0, path.dimension(vertex)), range(0, columns));
      auto after   = T_buf(_, range(0, path.dimension(vertex + 1)), range(0, columns));
      auto F       = vertex_matrix(path, vertex, action);
      for (int t = 0; t < r; ++t) after(t, _, _) = matmul(F, before(t, _, _));
      apply_kernel(after, path.backbone.vertices[vertex], action.kernel);
    }

    template <triqs_xca::diagram::Kind K>
    void integrate_left_edge(BlockDiagOpFun const &Gt, nda::array_view<dcomplex, 3> T_buf, diagram::BackbonePath<K> const &path, int edge,
                             int columns, Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 3> &GKt) {
      auto GKt_ep = GKt(_, range(0, path.dimension(edge + 1)), range(0, path.dimension(edge + 1)));
      GKt_ep      = Gt.get_block(path.edge_sector(edge));
      apply_kernel_product(GKt_ep, path.backbone.edges[edge].kernels, action.kernel);
      auto T_ep         = T_buf(_, range(0, path.dimension(edge + 1)), range(0, columns));
      auto const &itops = action.tau_mesh.dlr_it();
      T_ep              = itops.convolve(action.tau_mesh.beta(), itops.vals2coefs(GKt_ep), itops.vals2coefs(T_ep), cppdlr::TIME_ORDERED);
    }

    void multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, diagram::SelfEnergyBackbonePath const &path,
                                                    Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 4> &Tkaps,
                                                    nda::array<dcomplex, 3> &Tmu) {
      auto const r     = action.tau_mesh.size();
      int special      = path.backbone.endpoints.special_vertex;
      auto orientation = path.backbone.endpoint_orientation();
      auto signs       = path.backbone.endpoints.signs;
      auto T_v         = T_buf(_, range(0, path.dimension(special)), range(0, path.dimension(1)));
      auto T_vp        = T_buf(_, range(0, path.dimension(special + 1)), range(0, path.dimension(0)));
      auto Tkaps_v     = Tkaps(_, _, range(0, path.dimension(special)), range(0, path.dimension(0)));
      auto Tmu_v       = Tmu(_, range(0, path.dimension(special)), range(0, path.dimension(0)));

      for (int kap = 0; kap < path.zero_operators.size(); ++kap) {
        int flavour = path.zero_operators[kap];
        auto F_kap  = endpoint_matrix(flavour, orientation.early_operator(flavour).adjoint, path.sector(0), action);
        for (int t = 0; t < r; ++t) Tkaps_v(kap, t, _, _) = matmul(T_v(t, _, _), F_kap);
      }
      T_vp = 0;
      for (int mu : path.special_operators) {
        auto coefficients = endpoint_coefficients(mu, orientation.reflected(), action);
        auto F_mu         = endpoint_matrix(mu, orientation.late_operator(mu).adjoint, path.sector(special), action);
        Tmu_v             = 0;
        for (int kap = 0; kap < path.zero_operators.size(); ++kap) {
          int early        = path.zero_operators[kap];
          auto coefficient = coefficients(_, early);
          int sign         = signs(action.statistics[early], action.statistics[mu]) * orientation.reflection_sign();
          for (int t = 0; t < r; ++t) Tmu_v(t, _, _) += sign * coefficient(t) * Tkaps_v(kap, t, _, _);
        }
        for (int t = 0; t < r; ++t) T_vp(t, _, _) += matmul(F_mu, Tmu_v(t, _, _));
      }
    }

    // A nonclosing path can be omitted only when its endpoint coefficients are exactly zero.
    bool select_diagonal_self_energy(diagram::SelfEnergyBackbonePath const &path, Action<BlockOpSymQuartet> const &action) {
      if (path.sector(path.range.vertex_end) == path.sector(0)) return true;
      // Inspect exact input coefficients before arithmetic. No magnitude cutoff or
      // cancellation between distinct contributions is assumed here.
      bool reflected = path.backbone.endpoint_orientation().reflected();
      for (int late : path.special_operators) {
        auto coefficients = endpoint_coefficients(late, reflected, action);
        for (int early : path.zero_operators) {
          auto values = coefficients(_, early);
          if (std::any_of(values.begin(), values.end(), [](dcomplex z) { return z != dcomplex{}; }))
            throw std::invalid_argument("block_sparse self-energy: path from sector " + std::to_string(path.sector(0))
                                    + " ends in sector " + std::to_string(path.sector(path.range.vertex_end))
                                    + "; this partition does not close the indexed self-energy contribution. "
                                      "Use a coarser Hilbert-space partition for block-diagonal output.");
        }
      }
      return false;
    }

    // The returned active view borrows T and is consumed before scratch is reused.
    nda::array_const_view<dcomplex, 3> eval_self_energy_contribution(BlockDiagOpFun const &Gt, diagram::SelfEnergyBackbonePath const &path,
                                                                     double kernel_prefactor, Action<BlockOpSymQuartet> const &action,
                                                                     nda::array<dcomplex, 3> &T, nda::array<dcomplex, 3> &GKt,
                                                                     nda::array<dcomplex, 4> &Tkaps, nda::array<dcomplex, 3> &Tmu) {
      auto const &backbone = path.backbone;
      int m = backbone.order(), special = backbone.endpoints.special_vertex;

      T(_, range(0, path.dimension(1)), range(0, path.dimension(1))) = Gt.get_block(path.edge_sector(0));
      for (int v = 1; v < special; ++v) {
        multiply_left_vertex(T, path, v, path.dimension(1), action);
        integrate_left_edge(Gt, T, path, v, path.dimension(1), action, GKt);
      }

      multiply_left_vertex_and_right_zero_vertex(T, path, action, Tkaps, Tmu);

      for (int v = special + 1; v < 2 * m; ++v) {
        integrate_left_edge(Gt, T, path, v - 1, path.dimension(0), action, GKt);
        multiply_left_vertex(T, path, v, path.dimension(0), action);
      }

      T *= kernel_prefactor;
      auto result = T(_, range(0, path.dimension(2 * m)), range(0, path.dimension(0)));
      result *= backbone.self_energy_sign();
      return result;
    }

    // Accumulate into a separately owned output; propagator and scratch must not overlap it.
    void eval_self_energy(BlockDiagOpFun &Sigma, BlockDiagOpFun const &Gt, diagram::SelfEnergyBackbonePath const &path, double kernel_prefactor,
                          Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 3> &T, nda::array<dcomplex, 3> &GKt,
                          nda::array<dcomplex, 4> &Tkaps, nda::array<dcomplex, 3> &Tmu) {
      if (path.zero_operators.empty() || path.special_operators.empty()) return;
      if (!select_diagonal_self_energy(path, action)) return;
      auto result = eval_self_energy_contribution(Gt, path, kernel_prefactor, action, T, GKt, Tkaps, Tmu);
      Sigma.add_block(path.sector(0), result);
    }

    void eval_self_energy(BlockDiagOpFun &Sigma, BlockDiagOpFun const &Gt, triqs_xca::diagram::SelfEnergyBackbone const &backbone,
                          Action<BlockOpSymQuartet> const &action, OperatorSupportTable const &support, nda::array<dcomplex, 3> &T,
                          nda::array<dcomplex, 3> &GKt, nda::array<dcomplex, 4> &Tkaps, nda::array<dcomplex, 3> &Tmu) {
      double kernel_prefactor = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);
      triqs_xca::block_sparse::visit_self_energy_paths(Gt, backbone, action.operators, support, [&](diagram::SelfEnergyBackbonePath const &path) {
        eval_self_energy(Sigma, Gt, path, kernel_prefactor, action, T, GKt, Tkaps, Tmu);
      });
    }

    void multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, diagram::CorrelatorBackbonePath const &path, int vertex,
                               Action<BlockOpSymQuartet> const &action) {
      auto const r = action.tau_mesh.size();
      int rows     = path.dimension(path.range.vertex_end);
      auto before  = U_buf(_, range(0, rows), range(0, path.dimension(vertex)));
      auto after   = U_buf(_, range(0, rows), range(0, path.dimension(vertex + 1)));
      auto F       = vertex_matrix(path, vertex, action);
      for (int t = 0; t < r; ++t) before(t, _, _) = matmul(after(t, _, _), F);
      apply_kernel(before, path.backbone.vertices[vertex], action.kernel, true);
    }

    void integrate_right_edge(BlockDiagOpFun const &Gt, nda::array_view<dcomplex, 3> U_buf, diagram::CorrelatorBackbonePath const &path, int edge,
                              Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 3> &GKt) {
      int rows    = path.dimension(path.range.vertex_end);
      auto GKt_ep = GKt(_, range(0, path.dimension(edge + 1)), range(0, path.dimension(edge + 1)));
      GKt_ep      = Gt.get_block(path.edge_sector(edge));
      apply_kernel_product(GKt_ep, path.backbone.edges[edge].kernels, action.kernel);
      auto U_e          = U_buf(_, range(0, rows), range(0, path.dimension(edge + 1)));
      auto const &itops = action.tau_mesh.dlr_it();
      U_e               = itops.convolve(action.tau_mesh.beta(), itops.vals2coefs(U_e), itops.vals2coefs(GKt_ep), cppdlr::TIME_ORDERED);
    }

    nda::array_view<dcomplex, 3> eval_backbone_forward(BlockDiagOpFun const &Gt, diagram::CorrelatorBackbonePath const &path,
                                                       Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 3> &T,
                                                       nda::array<dcomplex, 3> &GKt) {
      auto const &backbone = path.backbone;
      auto const &interval = path.range;
      if (interval.empty()) throw std::invalid_argument("cannot evaluate an empty backbone segment");
      int columns  = path.dimension(interval.first_vertex);
      auto initial = T(_, range(0, columns), range(0, columns));
      initial      = Gt.get_block(path.edge_sector(interval.first_edge));
      apply_kernel_product(initial, backbone.edges[interval.first_edge].kernels, action.kernel);
      for (int v = interval.first_vertex; v < interval.vertex_end; ++v) {
        multiply_left_vertex(T, path, v, columns, action);
        integrate_left_edge(Gt, T, path, v, columns, action, GKt);
      }
      return T(_, range(0, path.dimension(interval.vertex_end)), range(0, columns));
    }

    nda::array_view<dcomplex, 3> eval_backbone_backward(BlockDiagOpFun const &Gt, diagram::CorrelatorBackbonePath const &path,
                                                        Action<BlockOpSymQuartet> const &action, nda::array<dcomplex, 3> &U,
                                                        nda::array<dcomplex, 3> &GKt) {
      auto const &backbone = path.backbone;
      auto const &interval = path.range;
      if (interval.empty()) throw std::invalid_argument("cannot evaluate an empty backbone segment");
      int rows     = path.dimension(interval.vertex_end);
      auto initial = U(_, range(0, rows), range(0, rows));
      initial      = Gt.get_block(path.edge_sector(interval.edge_end - 1));
      apply_kernel_product(initial, backbone.edges[interval.edge_end - 1].kernels, action.kernel);
      for (int v = interval.vertex_end - 1; v >= interval.first_vertex; --v) {
        multiply_right_vertex(U, path, v, action);
        integrate_right_edge(Gt, U, path, v - 1, action, GKt);
      }
      return U(_, range(0, rows), range(0, path.dimension(interval.first_vertex)));
    }

    void eval_correlator_completion(diagram::CorrelatorBackbonePath const &path, std::vector<BlockOp> const &mu_ops,
                                    std::vector<BlockOp> const &kap_ops, nda::array_const_view<dcomplex, 3> left,
                                    nda::array_const_view<dcomplex, 3> right, nda::array_view<dcomplex, 3> scratch,
                                    nda::array_view<dcomplex, 3> correlator) {
      auto const r = correlator.extent(0);
      if (path.zero_operators.empty() || path.special_operators.empty()) return;
      int mu       = path.special_operators.front();
      auto F_mu    = mu_ops[mu].get_block(path.sector(path.backbone.endpoints.special_vertex));
      auto product = scratch(_, range(0, left.extent(1)), range(0, right.extent(2)));
      for (int t = 0; t < r; ++t) product(t, _, _) = matmul(left(t, _, _), matmul(F_mu, right(t, _, _)));
      for (int kap : path.zero_operators) {
        auto F_kap = kap_ops[kap].get_block(path.sector(0));
        for (int t = 0; t < r; ++t) correlator(t, mu, kap) += trace(matmul(product(t, _, _), F_kap));
      }
    }

    void eval_correlator(nda::array_view<dcomplex, 3> output, BlockDiagOpFun const &Gt, triqs_xca::diagram::CorrelatorBackbone const &backbone,
                         std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                         triqs_xca::diagram::CorrelatorStatistics correlator_statistics, Action<BlockOpSymQuartet> const &action,
                         OperatorSupportTable const &support, nda::array<dcomplex, 3> &T, nda::array<dcomplex, 3> &U, nda::array<dcomplex, 3> &GKt) {
      auto const r            = action.tau_mesh.size();
      double kernel_prefactor = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel);
      // Preserve the per-backbone sum and sign before accumulating into the caller's result.
      auto correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
      auto scratch    = nda::zeros<dcomplex>(T.shape());
      std::vector<diagram::CorrelatorBackbonePath> zero_to_tau_paths, tau_to_beta_paths;
      std::vector<nda::array<dcomplex, 3>> zero_to_tau_results, tau_to_beta_results;
      auto [zero_to_tau, tau_to_beta] = backbone.endpoint_ranges();
      visit_backbone_paths(Gt, backbone, action.operators, support, zero_to_tau, [&](diagram::CorrelatorBackbonePath const &path) {
        auto result = eval_backbone_forward(Gt, path, action, T, GKt);
        // Global factors belong to the completion, applied once to each cached right result.
        result *= kernel_prefactor;
        result *= backbone.direction_sign;
        zero_to_tau_paths.push_back(path);
        zero_to_tau_results.emplace_back(result);
      });
      visit_backbone_paths(Gt, backbone, action.operators, support, tau_to_beta, [&](diagram::CorrelatorBackbonePath const &path) {
        auto result = eval_backbone_backward(Gt, path, action, U, GKt);
        tau_to_beta_paths.push_back(path);
        // cppdlr::reflect reshapes its input; an active block in U can be strided.
        tau_to_beta_results.emplace_back(action.tau_mesh.dlr_it().reflect(nda::make_regular(result)));
      });
      triqs_xca::block_sparse::visit_correlator_paths(Gt, zero_to_tau_paths, tau_to_beta_paths, mu_ops, kap_ops,
                                                      [&](diagram::CorrelatorBackbonePath const &path, int before_result, int after_result) {
                                                        eval_correlator_completion(path, mu_ops, kap_ops, tau_to_beta_results[after_result],
                                                                                   zero_to_tau_results[before_result], scratch, correlator);
                                                      });
      auto signs = backbone.endpoints.signs;
      correlator *= signs(correlator_statistics.at_zero, correlator_statistics.at_tau);
      output += correlator;
    }

    void eval_correlator(nda::array_view<dcomplex, 3> output, BlockDiagOpFun const &Gt, diagram::CorrelatorBackbonePath const &path,
                         std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                         triqs_xca::diagram::CorrelatorStatistics correlator_statistics, Action<BlockOpSymQuartet> const &action,
                         nda::array<dcomplex, 3> &T, nda::array<dcomplex, 3> &U, nda::array<dcomplex, 3> &GKt) {
      auto const r = action.tau_mesh.size();
      // Apply the endpoint sign to this contribution before adding it to the caller's result.
      auto correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
      if (path.zero_operators.empty() || path.special_operators.empty()) return;

      auto [zero_to_tau, tau_to_beta] = path.backbone.endpoint_ranges();
      auto before                     = segment_path(path, zero_to_tau);
      auto after                      = segment_path(path, tau_to_beta);
      auto right                      = eval_backbone_forward(Gt, before, action, T, GKt);
      right *= tau_kernel_prefactor(path.backbone.kernel_prefactor, action.kernel);
      right *= path.backbone.direction_sign;
      // Forward and backward traversals use separate T/U storage. Reflecting U needs
      // a regular copy, just as in the cached execution route above.
      auto left    = action.tau_mesh.dlr_it().reflect(nda::make_regular(eval_backbone_backward(Gt, after, action, U, GKt)));
      auto scratch = nda::zeros<dcomplex>(T.shape());
      eval_correlator_completion(path, mu_ops, kap_ops, left, right, scratch, correlator);
      auto signs = path.backbone.endpoints.signs;
      correlator *= signs(correlator_statistics.at_zero, correlator_statistics.at_tau);
      output += correlator;
    }

  } // namespace

  DiagramEvaluator::DiagramEvaluator(Action<BlockOpSymQuartet> model) : action(std::move(model)), Sigma({}, {}) {
    auto const r             = action.tau_mesh.size();
    auto const max_dimension = std::max(0L, nda::max_element(action.block_dims));
    T                        = nda::zeros<dcomplex>(r, max_dimension, max_dimension);
    U                        = nda::zeros<dcomplex>(r, max_dimension, max_dimension);
    GKt                      = nda::zeros<dcomplex>(r, max_dimension, max_dimension);
    Tkaps                    = nda::zeros<dcomplex>(max_sym_set_size(action.operators), r, max_dimension, max_dimension);
    Tmu                      = nda::zeros<dcomplex>(r, max_dimension, max_dimension);
  }

  DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::vector_const_view<double> hyb_poles,
                                     nda::array_const_view<dcomplex, 3> hyb_coeffs, BlockOpSymQuartet const &Fq, int n_int)
     : DiagramEvaluator(
          make_action(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false), hyb_poles, hyb_coeffs, Fq, n_int)) {}

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad)
     : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad);

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad,
                                     std::vector<many_body_operator_complex> const &dynint_ops, nda::array_const_view<dcomplex, 3> dynint_coeffs)
     : DiagramEvaluator(make_action(tau_mesh, hyb_poles, hyb_coeffs, ad, dynint_ops, dynint_coeffs)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad,
                                              std::vector<many_body_operator_complex> const &dynint_ops,
                                              nda::array_const_view<dcomplex, 3> dynint_coeffs);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                              triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad,
                                              std::vector<many_body_operator_complex> const &dynint_ops,
                                              nda::array_const_view<dcomplex, 3> dynint_coeffs);

  void DiagramEvaluator::reset() {
    auto const r = action.tau_mesh.size();
    T            = 0;
    U            = 0;
    GKt          = 0;
    Tkaps        = 0;
    Tmu          = 0;
    for (int i = 0; i < Sigma.get_num_block_cols(); i++) {
      Sigma.set_block(i, nda::zeros<dcomplex>(r, Sigma.get_block_size(i), Sigma.get_block_size(i)));
    }
  }

  block_gf<dlr_imtime> DiagramEvaluator::compute_self_energy(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc, action);
    triqs_xca::diagram::topology::validate(topology);
    long count =
       triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::self_energy);
    auto [first, last] = triqs_xca::detail::local_work_range(count);

    // Allocate Sigma and set to zero
    Sigma = Gt;
    Sigma *= 0;

    for (long f_ix = first; f_ix < last; ++f_ix) {
      auto const backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, action, f_ix);
      eval_self_energy(Sigma, Gt, backbone, action, support_, T, GKt, Tkaps, Tmu);
      Sigma.set_zero_block_indices();
    }

    auto result = self_energy_result(Sigma, action.tau_mesh);
    reset();
    return result;
  }

  BlockDiagOpFun DiagramEvaluator::compute_self_energy(BlockDiagOpFun const &Gt, triqs_xca::diagram::SelfEnergyBackbone const &backbone) {
    check_propagator(Gt, action);
    auto result = Gt;
    result *= 0;

    eval_self_energy(result, Gt, backbone, action, support_, T, GKt, Tkaps, Tmu);
    result.set_zero_block_indices();

    reset();
    return result;
  }

  BlockDiagOpFun DiagramEvaluator::compute_self_energy(BlockDiagOpFun const &Gt, diagram::SelfEnergyBackbonePath const &path) {
    require_complete_path(path);
    check_propagator(Gt, action);
    if (!path.special_operators.empty() && path.zero_operators.size() > Tkaps.extent(0))
      throw std::invalid_argument("self-energy endpoint selection exceeds this CPU evaluator's symmetry-group capacity");
    auto result = Gt;
    result *= 0;
    if (!path.zero_operators.empty() && !path.special_operators.empty())
      eval_self_energy(result, Gt, path, tau_kernel_prefactor(path.backbone.kernel_prefactor, action.kernel), action, T, GKt, Tkaps, Tmu);
    result.set_zero_block_indices();
    reset();
    return result;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(BlockDiagOpFun const &Gt,
                                                                        triqs_xca::diagram::CorrelatorBackbone const &backbone,
                                                                        std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                                                                        triqs_xca::diagram::CorrelatorStatistics correlator_statistics) {
    check_propagator(Gt, action);
    auto result = nda::zeros<dcomplex>(action.tau_mesh.size(), mu_ops.size(), kap_ops.size());
    eval_correlator(result, Gt, backbone, mu_ops, kap_ops, correlator_statistics, action, support_, T, U, GKt);
    return result;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(BlockDiagOpFun const &Gt, diagram::CorrelatorBackbonePath const &path,
                                                                        std::vector<BlockOp> const &mu_ops, std::vector<BlockOp> const &kap_ops,
                                                                        triqs_xca::diagram::CorrelatorStatistics correlator_statistics) {
    require_complete_path(path);
    if (path.special_operators.size() > 1) throw std::invalid_argument("correlator path supports one tau observable");
    check_propagator(Gt, action);
    auto result = nda::zeros<dcomplex>(action.tau_mesh.size(), mu_ops.size(), kap_ops.size());
    eval_correlator(result, Gt, path, mu_ops, kap_ops, correlator_statistics, action, T, U, GKt);
    return result;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(block_gf_view<dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) {
    auto const r = action.tau_mesh.size();
    auto Gt      = native_propagator(G_ppsc, action);
    triqs_xca::diagram::topology::validate(topology);
    auto mu_ops                        = single_particle_operators(false, action);
    auto kap_ops                       = single_particle_operators(true, action);
    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
    long count =
       triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::correlator);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    for (long f_ix = first; f_ix < last; ++f_ix) {
      auto const backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(topology, action, f_ix);
      eval_correlator(correlator, Gt, backbone, mu_ops, kap_ops, {}, action, support_, T, U, GKt);
    }
    return correlator;
  }

  template <bool isComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(block_gf_view<dlr_imtime> G_ppsc, std::vector<many_body_operator_complex> const &ops_tau,
                                                std::vector<many_body_operator_complex> const &ops_0,
                                                triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology) {
    auto const r = action.tau_mesh.size();

    // the statistics is the fermion parity of the operators, not whether they commute, see operator_statistics.hpp
    bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

    auto Gt = native_propagator(G_ppsc, action);
    triqs_xca::diagram::topology::validate(topology);

    // the operator blocks are placed by the subspaces of ad, which have to be those of the evaluator
    auto ad_dims = atom_block_dims(ad);
    bool same    = ad_dims.size() == action.block_dims.size();
    for (long b = 0; same && b < ad_dims.size(); ++b) same = action.block_dims(b) < 0 || ad_dims(b) == action.block_dims(b);
    if (!same)
      throw std::invalid_argument("compute_one_time_correlator: the subspaces of the atom_diag differ from those the evaluator was built for");

    auto mu_ops  = operator_matrices(ops_tau, ad);
    auto kap_ops = operator_matrices(ops_0, ad);

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());

    long count =
       triqs_xca::diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), triqs_xca::diagram::Kind::correlator);
    auto [first, last] = triqs_xca::detail::local_work_range(count);
    auto statistics    = triqs_xca::diagram::uniform_correlator_statistics(is_fermionic);
    for (long f_ix = first; f_ix < last; ++f_ix) {
      auto const backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(topology, action, f_ix);
      eval_correlator(correlator, Gt, backbone, mu_ops, kap_ops, statistics, action, support_, T, U, GKt);
    }

    return correlator;
  }

  template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(block_gf_view<dlr_imtime> G_ppsc,
                                                                                 std::vector<many_body_operator_complex> const &ops_tau,
                                                                                 std::vector<many_body_operator_complex> const &ops_0,
                                                                                 triqs::atom_diag::atom_diag<false> const &ad,
                                                                                 nda::array_const_view<int, 2> topology);

  template nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(block_gf_view<dlr_imtime> G_ppsc,
                                                                                 std::vector<many_body_operator_complex> const &ops_tau,
                                                                                 std::vector<many_body_operator_complex> const &ops_0,
                                                                                 triqs::atom_diag::atom_diag<true> const &ad,
                                                                                 nda::array_const_view<int, 2> topology);

} // namespace triqs_xca::block_sparse
