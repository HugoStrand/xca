#include "../block_sparse_common/fixture.hpp"
#include "../block_sparse_common/call_runner.hpp"
#include "../block_sparse_common/work_inventory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <tuple>

namespace {
  using namespace block_sparse_bench;
  using clock_type = std::chrono::steady_clock;
  double ms(clock_type::time_point begin) { return std::chrono::duration<double, std::milli>(clock_type::now() - begin).count(); }
  using Shape = std::tuple<std::string, long, long, long>;
} // namespace
int main(int argc, char **argv) {
  try {
    int n = 4, order = 3, samples = 3;
    long first = 0, count = 32, rank = 26;
    double lambda = 40.;
    bool analytic = false, dynamic = false, physical_mask = false;
    std::string partition = "number", topology_name = "crossing", shapes_file, work_file;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--analytic") {
        analytic = true;
        continue;
      }
      if (key == "--dynamic") {
        dynamic = true;
        continue;
      }
      if (key == "--physical-mask") {
        physical_mask = true;
        continue;
      }
      if (i + 1 == argc) throw std::invalid_argument("Missing argument for " + key);
      std::string value = argv[++i];
      if (key == "--modes")
        n = std::stoi(value);
      else if (key == "--partition")
        partition = value;
      else if (key == "--topology")
        topology_name = value;
      else if (key == "--order")
        order = std::stoi(value);
      else if (key == "--first")
        first = std::stol(value);
      else if (key == "--count")
        count = std::stol(value);
      else if (key == "--rank")
        rank = std::stol(value);
      else if (key == "--lambda")
        lambda = std::stod(value);
      else if (key == "--samples")
        samples = std::stoi(value);
      else if (key == "--shapes")
        shapes_file = value;
      else if (key == "--work")
        work_file = value;
      else
        throw std::invalid_argument("Unknown argument " + key);
    }
    if (n < 2 || n > 14 || (!analytic && n > 10) || order < 1 || order > 12 || count < 0 || count > 4096 || first < 0 || rank < 1 || samples < 1
        || samples > 7)
      throw std::invalid_argument("Invalid/beyond-bounded inventory request (actual atom_diag limited to 10 modes)");
    if (!std::isfinite(lambda) || lambda < 10. || lambda > 10000.) throw std::invalid_argument("Invalid DLR Lambda");
    if (partition != "number" && partition != "occupation" && partition != "mixed") throw std::invalid_argument("Unknown partition");
    if (topology_name != "crossing" && topology_name != "chain") throw std::invalid_argument("Unknown topology");
    if (analytic && dynamic) throw std::invalid_argument("Analytic inventory does not infer dynamic-channel connectivity");
    if (analytic && !work_file.empty()) throw std::invalid_argument("Analytic inventory has no executed path work");
    if (analytic && physical_mask) throw std::invalid_argument("Analytic inventory has no physical propagator mask");
    auto start              = clock_type::now();
    auto dims               = analytic_dimensions(n, partition);
    std::size_t maximum_set = partition == "occupation" ? 1 : partition == "number" ? n : (n + 1) / 2;
    std::size_t sets        = partition == "occupation" ? n : partition == "number" ? 1 : 2;
    std::size_t owned = 0, bound = 0, payload = 0, nnz = 0;
    long last        = first;
    long live_blocks = dims.size();
    gpu::PathCounts counts;
    std::map<Shape, std::size_t> shapes;
    double setup = 0, trace = 0;
    if (!analytic) {
      auto ad    = atom(n, partition);
      auto coeff = coefficients(ad);
      std::vector<many_body_operator_complex> interactions;
      if (dynamic) {
        many_body_operator_complex Q;
        for (int i = 0; i < n; ++i) Q += triqs::operators::n("f", i);
        interactions = {Q, triqs::operators::c_dag("f", 0) * triqs::operators::c("f", 1)};
      }
      auto dc = nda::zeros<dcomplex>(2, interactions.size(), interactions.size());
      for (long p = 0; p < 2; ++p)
        for (long i = 0; i < long(interactions.size()); ++i) dc(p, i, i) = .03 - .01 * p;
      auto model  = gpu::export_model(ad, coeff, interactions, dc);
      dims        = model.dimensions();
      live_blocks = dims.size();
      sets        = model.sets().size();
      maximum_set = 0;
      for (auto const &set : model.sets()) maximum_set = std::max(maximum_set, set.size());
      owned   = model.host_storage_bytes();
      bound   = model.host_storage_bound_bytes();
      payload = model.sparse_payload_bytes();
      for (int i = 0; i < model.num_fields(); ++i) nnz = gpu::checked_add(nnz, model.field(i).nonzeros());
      triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, lambda / 2., 1e-10, false);
      rank = mesh.dlr_it().rank();
      WorkInventory work(work_file, rank);
      nda::vector<double> poles{-1.5, .6};
      nda::array<int, 2> topology(order, 2);
      for (int i = 0; i < order; ++i) {
        topology(i, 0) = i;
        topology(i, 1) = i + order;
      }
      if (topology_name == "chain" && order > 1) {
        topology(0, 0) = 0;
        topology(0, 1) = 2;
        for (int i = 1; i < order - 1; ++i) {
          topology(i, 0) = 2 * i - 1;
          topology(i, 1) = 2 * i + 2;
        }
        topology(order - 1, 0) = 2 * order - 3;
        topology(order - 1, 1) = 2 * order - 1;
      }
      Backbone backbone(topology, model.num_fields(), model.num_interactions());
      long total = backbone.num_flat_indices(poles.size());
      if (first > total) throw std::out_of_range("First index exceeds domain");
      last = first + std::min(count, total - first);
      std::vector<bool> live(dims.size(), true); // Explicit all-live metadata control.
      if (physical_mask) {
        auto G      = propagator(ad, mesh);
        live_blocks = 0;
        for (long b = 0; b < G.size(); ++b) {
          live[b] = nda::max_element(nda::abs(G[b].data())) >= 1e-16;
          live_blocks += live[b];
        }
      }
      setup    = ms(start);
      auto run = [&](bool record) {
        return gpu::for_each_self_energy_path(
           model, poles, topology, live, first, last, [&](Backbone &selected, triqs_xca::block_sparse::SelfEnergyPath const &path) {
             if (!record) return;
             work.record(model, selected, path);
             auto add = [&](std::string kind, long M, long N, long K, std::size_t operations = 1) {
               auto &frequency = shapes[{kind, M, N, K}];
               frequency       = gpu::checked_add(frequency, operations);
             };
             long s = path.special_vertex;
             for (int v = 1; v < 2 * order; ++v) {
               if (v == s) continue;
               long anchor = path.dimensions[v < s ? 1 : 0];
               add("vertex", path.dimensions[v + 1], anchor, path.dimensions[v]);
             }
             for (int e = 1; e < 2 * order - 1; ++e) {
               long M = path.dimensions[e + 1], anchor = path.dimensions[e < s ? 1 : 0];
               add("convolution", M, anchor, M, 3); // Three GEMMs per DLR node.
               for (auto rhs : {gpu::checked_multiply(M, M), gpu::checked_multiply(M, anchor)}) {
                 if (rhs > std::size_t(std::numeric_limits<long>::max())) throw std::overflow_error("RHS inventory overflow");
                 add("solve_rhs", rank, rhs, rank);
               }
             }
             auto q = model.sets()[path.zero_set].size();
             add("special_right", path.dimensions[s], path.dimensions[0], path.dimensions[1], gpu::checked_multiply(q, q));
             add("special_left", path.dimensions[s + 1], path.dimensions[0], path.dimensions[s], q);
             add("special_mix", path.dimensions[s], path.dimensions[0], q, q);
           });
      };
      counts = run(true);
      (void)run(false); // Warm scheduling; no numerical arithmetic or CUDA.
      start = clock_type::now();
      for (int i = 0; i < samples; ++i) (void)run(false);
      trace = ms(start) / samples;
    }
    auto memory = gpu::inventory_memory(rank, dims, maximum_set);
    std::cout
       << "analytic,partition,modes,dynamic_channels,topology,rank,blocks,max_dim,symmetry_sets,max_set_size,bare_nnz,sparse_payload_bytes,model_host_bytes,model_bound_bytes,G_bytes,max_rank_bytes,legacy_scratch_bytes,legacy_correlator_sides_bytes,order,first,last,proposed,accepted,pruned,setup_ms,trace_ms,mask_policy,live_blocks\n";
    std::cout << analytic << ',' << partition << ',' << n << ',' << (dynamic ? 2 : 0) << ',' << topology_name << ',' << rank << ',' << dims.size()
              << ',' << *std::max_element(dims.begin(), dims.end()) << ',' << sets << ',' << maximum_set << ',' << nnz << ',' << payload << ','
              << owned << ',' << bound << ',' << memory.function << ',' << memory.maximum_rank_buffer << ',' << memory.legacy_scratch << ','
              << memory.legacy_correlator_sides << ',' << order << ',' << first << ',' << last << ',' << counts.proposed << ',' << counts.accepted
              << ',' << counts.pruned << ',' << setup << ',' << trace << ',' << (physical_mask ? "atomic_G0" : "all_live") << ',' << live_blocks
              << '\n';
    if (!shapes_file.empty()) {
      std::ofstream stream(shapes_file);
      if (!stream) throw std::runtime_error("Cannot write shape inventory");
      stream << "kind,M,N,K,calls,rank\n";
      for (auto const &[shape, frequency] : shapes) {
        auto const &[kind, M, N, K] = shape;
        stream << kind << ',' << M << ',' << N << ',' << K << ',' << frequency << ',' << rank << '\n';
      }
    }
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
