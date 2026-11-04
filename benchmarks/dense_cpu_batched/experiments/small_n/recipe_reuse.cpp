// Untimed reuse census for the exact model and sampling of dense_cpu_batched.
// This experiment deliberately includes the parent benchmark to share its fixture.
#define main dense_cpu_batched_original_main
#include "../../main.cpp"
#undef main

#include <numeric>
#include <set>
#include <sstream>

namespace {
  std::string edge_key(diagram::Edge const &edge) {
    std::ostringstream out;
    for (auto const &factor : edge.kernels) out << factor.pole_index << ':' << factor.argument_sign << ';';
    return out.str();
  }

  // Only fields actually consumed by get_operator/apply_kernel enter the key.
  std::string vertex_key(diagram::Vertex const &vertex) {
    std::ostringstream out;
    out << vertex.flavour << ':' << vertex.barred << ':' << vertex.adjoint;
    if (vertex.barred) out << ":p" << vertex.pole_index;
    if (vertex.kernel_sign) out << ":k" << vertex.pole_index << ':' << vertex.kernel_sign;
    return out.str();
  }

  template <diagram::Kind K>
  void census(std::string const &kind, int order, std::string const &selection,
              std::vector<diagram::Backbone<K>> const &backbones, long total_choices) {
    // Each entry below corresponds to an actual convolution call, in execution order.
    std::vector<std::vector<std::string>> edge_recipes, prefixes;
    for (auto const &backbone : backbones) {
      auto ranges = backbone.endpoint_ranges();
      std::vector<std::string> edges, prefix_steps;
      std::string prefix = "G";
      for (int v = ranges.before_special.first_vertex; v < ranges.before_special.vertex_end; ++v) {
        auto recipe = edge_key(backbone.edges[v]);
        edges.push_back(recipe);
        prefix += "V{" + vertex_key(backbone.vertices[v]) + "}E{" + recipe + "}";
        prefix_steps.push_back(prefix);
      }
      if constexpr (K == diagram::Kind::self_energy) {
        for (int v = ranges.after_special.first_vertex; v < ranges.after_special.vertex_end; ++v)
          edges.push_back(edge_key(backbone.edges[v - 1]));
      } else {
        for (int v = ranges.after_special.vertex_end - 1; v >= ranges.after_special.first_vertex; --v)
          edges.push_back(edge_key(backbone.edges[v - 1]));
      }
      edge_recipes.push_back(std::move(edges));
      prefixes.push_back(std::move(prefix_steps));
    }
    std::set<std::string> global_edges, global_prefixes;
    long calls = 0, empty_calls = 0, prefix_calls = 0, batch_position_unique = 0, batch_any_unique = 0;
    for (std::size_t begin = 0; begin < backbones.size(); begin += 8) {
      auto end = std::min(begin + 8, backbones.size());
      std::set<std::string> all_batch;
      for (std::size_t e = 0; e < edge_recipes.front().size(); ++e) {
        std::set<std::string> position;
        for (auto b = begin; b < end; ++b) {
          auto const &recipe = edge_recipes[b][e];
          position.insert(recipe);
          all_batch.insert(recipe);
          global_edges.insert(recipe);
          ++calls;
          empty_calls += recipe.empty();
        }
        batch_position_unique += position.size();
      }
      batch_any_unique += all_batch.size();
    }
    for (auto const &sequence : prefixes)
      for (auto const &prefix : sequence) {
        global_prefixes.insert(prefix);
        ++prefix_calls;
      }
    auto hit = [](long count, long unique) { return count ? 1. - double(unique) / count : 0.; };
    std::cout << kind << ',' << order << ',' << selection << ',' << total_choices << ',' << backbones.size() << ',' << calls << ','
              << global_edges.size() << ',' << double(empty_calls) / calls << ',' << hit(calls, global_edges.size()) << ','
              << hit(calls, batch_position_unique) << ',' << hit(calls, batch_any_unique) << ',' << prefix_calls << ','
              << global_prefixes.size() << ',' << hit(prefix_calls, global_prefixes.size()) << '\n';
  }
}

int main() try {
  auto m = model(false); // Recipe identity is independent of DLR rank.
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  std::cout << std::setprecision(10);
  std::cout << "# Exact ordered kernel lists; no canonicalization or numerical zero pruning. Batch capacity 8.\n";
  std::cout << "# Global hits assume unbounded call-local caching. Prefixes count only completed pre-endpoint steps, excluding initial G.\n";
  std::cout << "# Consecutive windows have the same <=63 count as the sampled call; midpoint and last windows check ordering bias.\n";
  std::cout << "kind,order,selection,total_choices,backbones,convolution_calls,unique_edge_recipes,empty_edge_fraction,edge_global_hit_fraction,edge_batch8_position_hit_fraction,edge_batch8_any_position_hit_fraction,initial_prefix_steps,unique_initial_prefix_steps,initial_prefix_global_hit_fraction\n";
  for (int order : {2, 3, 5}) {
    auto topology = crossing_topology(order);
    for (std::string const kind : {"self_energy", "pairs", "correlator"}) {
      bool pairs = kind == "pairs", correlator = kind == "correlator";
      long count = pairs ? diagram::count_self_energy_pairs(order, action.n_flavours(), action.poles.size()) :
                          diagram::count_diagrams(order, action.n_flavours(), action.poles.size(),
                                                  correlator ? diagram::Kind::correlator : diagram::Kind::self_energy);
      for (std::string const selection : {"sampled", "first", "middle", "last"}) {
        auto indices = sampled_indices(count);
        if (selection != "sampled") {
          long first = selection == "first" ? 0 : selection == "last" ? count - indices.size() : (count - indices.size()) / 2;
          std::iota(indices.begin(), indices.end(), first);
        }
        std::vector<diagram::SelfEnergyBackbone> se;
        std::vector<diagram::CorrelatorBackbone> corr;
        for (long index : indices) {
          if (correlator) corr.push_back(diagram::prepare_backbone<diagram::Kind::correlator>(topology, action, index));
          else if (pairs) se.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(topology, diagram::decode_self_energy_pair_arcs(order, action, index)));
          else se.push_back(diagram::prepare_backbone<diagram::Kind::self_energy>(topology, action, index));
        }
        if (correlator) census(kind, order, selection, corr, count);
        else census(kind, order, selection, se, count);
      }
    }
  }
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
