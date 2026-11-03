#include "diagram_evaluator_batched.hpp"
#include "triqs_xca/work_partition.hpp"
#include <nda/blas/gemm.hpp>
#include <algorithm>
#include <stdexcept>

namespace triqs_xca::dense {
  using nda::dcomplex;
  using nda::range;
  using triqs::gfs::block_gf;
  using triqs::gfs::block_gf_view;
  using triqs::mesh::dlr_imtime;
  constexpr auto all = range::all;

  namespace {
    long checked_capacity(long capacity) {
      if (capacity <= 0) throw std::invalid_argument("BatchedDiagramEvaluator: batch size must be positive");
      return capacity;
    }

    void check_propagator(nda::array_const_view<dcomplex, 3> G, Action<FSet> const &action) {
      if (G.shape() != std::array<long, 3>{action.tau_mesh.size(), action.block_dims(0), action.block_dims(0)})
        throw std::invalid_argument("BatchedDiagramEvaluator: propagator must have shape (DLR rank, N, N)");
    }

    void check_observables(nda::array_const_view<dcomplex, 3> mu_ops, nda::array_const_view<dcomplex, 3> kap_ops, Action<FSet> const &action) {
      for (auto operators : {mu_ops, kap_ops})
        if (operators.extent(1) != action.block_dims(0) || operators.extent(2) != action.block_dims(0))
          throw std::invalid_argument("BatchedDiagramEvaluator: observable operators must have shape (count, N, N)");
    }

    nda::array_const_view<dcomplex, 3> native_propagator(block_gf_view<dlr_imtime> G, Action<FSet> const &action) {
      if (G.size() != 1) throw std::invalid_argument("BatchedDiagramEvaluator: propagator must have a single dense block");
      check_propagator(G[0].data(), action);
      return G[0].data();
    }

    nda::array_view<dcomplex, 4> active(nda::array<dcomplex, 4> &buffer, long batch) {
      return {{buffer.extent(0), batch, buffer.extent(2), buffer.extent(3)}, buffer.data()};
    }

    template <diagram::Kind K> std::span<diagram::Backbone<K> const> next_batch(std::span<diagram::Backbone<K> const> pending, long capacity) {
      long count        = 1;
      auto const &first = pending.front();
      while (count < std::min<long>(capacity, pending.size())) {
        auto const &next = pending[count];
        if (next.vertices.size() != first.vertices.size() || next.edges.size() != first.edges.size()
            || next.endpoints.special_vertex != first.endpoints.special_vertex)
          break;
        ++count;
      }
      return pending.first(count);
    }

    template <diagram::Kind K>
    void initialize(nda::array<dcomplex, 4> &T, nda::array_const_view<dcomplex, 3> G, std::span<diagram::Backbone<K> const> batch) {
      auto values = active(T, batch.size());
      for (long b = 0; b < batch.size(); ++b) values(all, b, all, all) = G;
    }

    template <diagram::Kind K>
    void multiply_vertex(nda::array<dcomplex, 4> &T, nda::array<dcomplex, 4> &scratch, std::span<diagram::Backbone<K> const> batch, int vertex_index,
                         Action<FSet> const &action, bool right = false) {
      auto input = active(T, batch.size()), output = active(scratch, batch.size());
      for (long b = 0; b < batch.size(); ++b) {
        auto const &vertex = batch[b].vertices[vertex_index];
        auto F             = action.operators.get_operator(vertex);
        for (long t = 0; t < input.extent(0); ++t) {
          if (right)
            nda::blas::gemm(1., input(t, b, all, all), F, 0., output(t, b, all, all));
          else
            nda::blas::gemm(1., F, input(t, b, all, all), 0., output(t, b, all, all));
        }
        apply_kernel(output(all, b, all, all), vertex, action.kernel, right);
      }
      std::swap(T, scratch);
    }

    template <diagram::Kind K>
    void integrate_edge(nda::array<dcomplex, 4> &T, nda::array<dcomplex, 4> &GKt, nda::array_const_view<dcomplex, 3> G,
                        std::span<diagram::Backbone<K> const> batch, int edge_index, Action<FSet> const &action,
                        cppdlr::fast_convolve const &convolution, cppdlr::fast_convolve_batched_workspace &work, bool right = false) {
      auto result = active(T, batch.size()), edge = active(GKt, batch.size());
      for (long b = 0; b < batch.size(); ++b) {
        edge(all, b, all, all) = G;
        apply_kernel_product(edge(all, b, all, all), batch[b].edges[edge_index].kernels, action.kernel);
      }
      if (right)
        convolution.convolve(action.tau_mesh.beta(), result, edge, result, work);
      else
        convolution.convolve(action.tau_mesh.beta(), edge, result, result, work);
    }

    template <diagram::Kind K>
    void initial_segment(nda::array<dcomplex, 4> &T, nda::array<dcomplex, 4> &GKt, nda::array_const_view<dcomplex, 3> G,
                         std::span<diagram::Backbone<K> const> batch, Action<FSet> const &action, cppdlr::fast_convolve const &convolution,
                         cppdlr::fast_convolve_batched_workspace &work) {
      initialize(T, G, batch);
      auto range = batch.front().endpoint_ranges().before_special;
      for (int v = range.first_vertex; v < range.vertex_end; ++v) {
        multiply_vertex(T, GKt, batch, v, action);
        integrate_edge(T, GKt, G, batch, v, action, convolution, work);
      }
    }

    void endpoint_sum(nda::array_view<dcomplex, 3> T, OperatorPairOrientation orientation, diagram::PermutationSigns const &signs,
                      Action<FSet> const &action, nda::array<dcomplex, 4> &Tkaps, nda::array<dcomplex, 3> &Tmu) {
      auto coefficients = orientation.reflected() ? action.tau_values_reflect() : action.tau_values();
      for (int kap = 0; kap < action.n_flavours(); ++kap) {
        auto F = action.operators.get_operator(orientation.early_operator(kap).adjoint, kap);
        for (long t = 0; t < T.extent(0); ++t) nda::blas::gemm(1., T(t, all, all), F, 0., Tkaps(kap, t, all, all));
      }
      T = 0;
      for (int mu = 0; mu < action.n_flavours(); ++mu) {
        Tmu = 0;
        for (int kap = 0; kap < action.n_flavours(); ++kap) {
          auto [row, column] = orientation.coefficient_indices(mu, kap);
          int sign           = signs(action.statistics[kap], action.statistics[mu]) * orientation.reflection_sign();
          for (long t = 0; t < T.extent(0); ++t) Tmu(t, all, all) += sign * coefficients(t, row, column) * Tkaps(kap, t, all, all);
        }
        auto F = action.operators.get_operator(orientation.late_operator(mu).adjoint, mu);
        for (long t = 0; t < T.extent(0); ++t) nda::blas::gemm(1., F, Tmu(t, all, all), 1., T(t, all, all));
      }
    }

    void eval_self_energy(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> G,
                          std::span<diagram::SelfEnergyBackbone const> batch, bool paired, Action<FSet> const &action, nda::array<dcomplex, 4> &T,
                          nda::array<dcomplex, 4> &U, nda::array<dcomplex, 4> &GKt, nda::array<dcomplex, 4> &Tkaps, nda::array<dcomplex, 3> &Tmu,
                          cppdlr::fast_convolve const &convolution, cppdlr::fast_convolve_batched_workspace &work) {
      initial_segment(T, GKt, G, batch, action, convolution, work);
      if (paired) active(U, batch.size()) = active(T, batch.size());
      for (long b = 0; b < batch.size(); ++b) {
        auto const &backbone = batch[b];
        auto values          = active(T, batch.size());
        endpoint_sum(values(all, b, all, all), backbone.endpoint_orientation(), backbone.endpoints.signs, action, Tkaps, Tmu);
        if (paired) {
          auto opposite = active(U, batch.size());
          endpoint_sum(opposite(all, b, all, all), backbone.endpoint_orientation().opposite(), backbone.endpoints.signs, action, Tkaps, Tmu);
          values(all, b, all, all) -= opposite(all, b, all, all);
        }
      }
      auto range = batch.front().endpoint_ranges().after_special;
      for (int v = range.first_vertex; v < range.vertex_end; ++v) {
        integrate_edge(T, GKt, G, batch, v - 1, action, convolution, work);
        multiply_vertex(T, GKt, batch, v, action);
      }
      auto values = active(T, batch.size());
      for (long b = 0; b < batch.size(); ++b) {
        double prefactor = tau_kernel_prefactor(batch[b].kernel_prefactor, action.kernel) * batch[b].self_energy_sign();
        result += prefactor * values(all, b, all, all);
      }
    }

    void eval_correlator(nda::array_view<dcomplex, 3> result, nda::array_const_view<dcomplex, 3> G,
                         std::span<diagram::CorrelatorBackbone const> batch, nda::array_const_view<dcomplex, 3> mu_ops,
                         nda::array_const_view<dcomplex, 3> kap_ops, diagram::CorrelatorStatistics statistics, Action<FSet> const &action,
                         nda::array<dcomplex, 4> &T, nda::array<dcomplex, 4> &U, nda::array<dcomplex, 4> &GKt, nda::array<dcomplex, 3> &Tmu,
                         cppdlr::fast_convolve const &convolution, cppdlr::fast_convolve_batched_workspace &work) {
      initial_segment(T, GKt, G, batch, action, convolution, work);
      initialize(U, G, batch);
      auto range = batch.front().endpoint_ranges().after_special;
      for (int v = range.vertex_end - 1; v >= range.first_vertex; --v) {
        multiply_vertex(U, GKt, batch, v, action, true);
        integrate_edge(U, GKt, G, batch, v - 1, action, convolution, work, true);
      }
      auto left = active(T, batch.size()), right = active(U, batch.size());
      // Reflection already accepts arbitrary trailing axes, sharing this transform too.
      right = action.tau_mesh.dlr_it().reflect(right);
      nda::matrix<dcomplex> product(G.extent(1), G.extent(1));
      for (long b = 0; b < batch.size(); ++b) {
        auto const &backbone = batch[b];
        double prefactor     = tau_kernel_prefactor(backbone.kernel_prefactor, action.kernel) * backbone.direction_sign;
        int sign             = backbone.endpoints.signs(statistics.at_zero, statistics.at_tau);
        for (long mu = 0; mu < mu_ops.extent(0); ++mu) {
          for (long t = 0; t < G.extent(0); ++t) {
            nda::blas::gemm(1., mu_ops(mu, all, all), left(t, b, all, all), 0., product);
            nda::blas::gemm(prefactor, right(t, b, all, all), product, 0., Tmu(t, all, all));
          }
          for (long kap = 0; kap < kap_ops.extent(0); ++kap)
            for (long t = 0; t < G.extent(0); ++t) {
              dcomplex trace = 0.;
              for (long i = 0; i < G.extent(1); ++i)
                for (long j = 0; j < G.extent(1); ++j) trace += Tmu(t, i, j) * kap_ops(kap, j, i);
              result(t, mu, kap) += sign * trace;
            }
        }
      }
    }

    template <diagram::Kind K, typename Prepare, typename Consume> void visit_topology(long count, long capacity, Prepare prepare, Consume consume) {
      auto [first, last] = triqs_xca::detail::local_work_range(count);
      std::vector<diagram::Backbone<K>> batch;
      batch.reserve(std::min(capacity, last - first));
      for (long begin = first; begin < last;) {
        batch.clear();
        long end = begin + std::min(capacity, last - begin);
        for (; begin < end; ++begin) batch.push_back(prepare(begin));
        consume(std::span<diagram::Backbone<K> const>{batch});
      }
    }
  } // namespace

  BatchedDiagramEvaluator::BatchedDiagramEvaluator(Action<FSet> model, long batch_size)
     : action(std::move(model)),
       capacity(checked_capacity(batch_size)),
       convolution(action.tau_mesh.dlr_it()),
       convolution_work(action.tau_mesh.size(), capacity, action.block_dims(0), action.block_dims(0), action.block_dims(0)),
       T(action.tau_mesh.size(), capacity, action.block_dims(0), action.block_dims(0)),
       U(T.shape()),
       GKt(T.shape()),
       Tkaps(action.n_flavours(), action.tau_mesh.size(), action.block_dims(0), action.block_dims(0)),
       Tmu(action.tau_mesh.size(), action.block_dims(0), action.block_dims(0)) {}

  nda::array<dcomplex, 3> BatchedDiagramEvaluator::compute_self_energy(nda::array_const_view<dcomplex, 3> G,
                                                                       std::span<diagram::SelfEnergyBackbone const> pending) {
    check_propagator(G, action);
    auto result = nda::zeros<dcomplex>(G.shape());
    while (!pending.empty()) {
      auto batch = next_batch(pending, capacity);
      eval_self_energy(result, G, batch, false, action, T, U, GKt, Tkaps, Tmu, convolution, convolution_work);
      pending = pending.subspan(batch.size());
    }
    return result;
  }

  nda::array<dcomplex, 3> BatchedDiagramEvaluator::compute_self_energy_by_pairs(nda::array_const_view<dcomplex, 3> G,
                                                                                std::span<diagram::SelfEnergyBackbone const> pending) {
    check_propagator(G, action);
    auto result = nda::zeros<dcomplex>(G.shape());
    while (!pending.empty()) {
      auto batch = next_batch(pending, capacity);
      eval_self_energy(result, G, batch, true, action, T, U, GKt, Tkaps, Tmu, convolution, convolution_work);
      pending = pending.subspan(batch.size());
    }
    return result;
  }

  nda::array<dcomplex, 3> BatchedDiagramEvaluator::compute_one_time_correlator(nda::array_const_view<dcomplex, 3> G,
                                                                               std::span<diagram::CorrelatorBackbone const> pending,
                                                                               nda::array_const_view<dcomplex, 3> mu_ops,
                                                                               nda::array_const_view<dcomplex, 3> kap_ops,
                                                                               diagram::CorrelatorStatistics statistics) {
    check_propagator(G, action);
    check_observables(mu_ops, kap_ops, action);
    auto result = nda::zeros<dcomplex>(G.extent(0), mu_ops.extent(0), kap_ops.extent(0));
    while (!pending.empty()) {
      auto batch = next_batch(pending, capacity);
      eval_correlator(result, G, batch, mu_ops, kap_ops, statistics, action, T, U, GKt, Tmu, convolution, convolution_work);
      pending = pending.subspan(batch.size());
    }
    return result;
  }

  block_gf<dlr_imtime> BatchedDiagramEvaluator::compute_self_energy(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology) {
    auto propagator = native_propagator(G, action);
    diagram::topology::validate(topology);
    auto result = nda::zeros<dcomplex>(propagator.shape());
    long count  = diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), diagram::Kind::self_energy);
    visit_topology<diagram::Kind::self_energy>(
       count, capacity, [&](long index) { return diagram::prepare_backbone<diagram::Kind::self_energy>(topology, action, index); },
       [&](auto batch) { result += compute_self_energy(propagator, batch); });
    return std::vector{triqs::gfs::gf<dlr_imtime>(action.tau_mesh, result)};
  }

  block_gf<dlr_imtime> BatchedDiagramEvaluator::compute_self_energy_by_pairs(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology) {
    auto propagator = native_propagator(G, action);
    diagram::topology::validate(topology);
    auto result = nda::zeros<dcomplex>(propagator.shape());
    long count  = diagram::count_self_energy_pairs(topology.extent(0), action.n_flavours(), action.poles.size());
    visit_topology<diagram::Kind::self_energy>(
       count, capacity,
       [&](long index) {
         return diagram::prepare_backbone<diagram::Kind::self_energy>(topology,
                                                                      diagram::decode_self_energy_pair_arcs(topology.extent(0), action, index));
       },
       [&](auto batch) { result += compute_self_energy_by_pairs(propagator, batch); });
    return std::vector{triqs::gfs::gf<dlr_imtime>(action.tau_mesh, result)};
  }

  nda::array<dcomplex, 3> BatchedDiagramEvaluator::compute_single_ptcle_gf(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology) {
    return compute_one_time_correlator(G, topology, action.operators.Fs(range(action.n_hyb), all, all),
                                       action.operators.F_dags(range(action.n_hyb), all, all));
  }

  nda::array<dcomplex, 3> BatchedDiagramEvaluator::compute_one_time_correlator(block_gf_view<dlr_imtime> G, nda::array_const_view<int, 2> topology,
                                                                               nda::array_const_view<dcomplex, 3> mu_ops,
                                                                               nda::array_const_view<dcomplex, 3> kap_ops,
                                                                               diagram::CorrelatorStatistics statistics) {
    auto propagator = native_propagator(G, action);
    diagram::topology::validate(topology);
    check_observables(mu_ops, kap_ops, action);
    auto result = nda::zeros<dcomplex>(propagator.extent(0), mu_ops.extent(0), kap_ops.extent(0));
    long count  = diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), diagram::Kind::correlator);
    visit_topology<diagram::Kind::correlator>(
       count, capacity, [&](long index) { return diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index); },
       [&](auto batch) { result += compute_one_time_correlator(propagator, batch, mu_ops, kap_ops, statistics); });
    return result;
  }
} // namespace triqs_xca::dense
