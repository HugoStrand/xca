#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_utils/evaluators.hpp"
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/block_sparse/bind/self_energy.hpp>
#include <triqs_xca/bind/tau_kernel.hpp>

using namespace triqs_xca;
namespace bs = triqs_xca::block_sparse;
namespace tu = triqs_xca::test_utils;
using Clock  = std::chrono::steady_clock;

long resident_kib() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.starts_with("VmRSS:")) return std::stol(line.substr(6));
  return -1; // RSS sampling is available on Linux only.
}

tu::EvalModel make_model(std::string const &name) {
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

int main(int argc, char **argv) try {
  std::string name = argc > 1 ? argv[1] : "unequal_sets_nA0";
  int order        = argc > 2 ? std::stoi(argv[2]) : 3;
  int repeats      = argc > 3 ? std::stoi(argv[3]) : 8;
  if (order < 1 || order > 12 || repeats < 1) throw std::invalid_argument("usage: backbone_cpu [model] [order 1..12] [positive repeats]");
  auto model = make_model(name);
  bs::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  auto bgf = tu::subspace_blocks(model.G, model.ad, model.mesh);
  bs::BlockDiagOpFun G(bgf);
  auto [operators, labels] = bs::dynint::get_operators_and_interactions(model.ad, model.hyb_coeffs, model.dynint_coeffs, model.dynint_ops);
  bs::OperatorSupportTable support(operators);
  auto arc_catalog = triqs_xca::diagram::hybridization_retarded_catalog(model.n_hyb(), model.n_int());
  nda::array<int, 2> arcs(order, 2);
  for (int a = 0; a < order; ++a) arcs(a, nda::range::all) = nda::vector<int>{a, a + order};
  triqs_xca::diagram::Topology topology{arcs};
  long count   = triqs_xca::diagram::count_diagrams(order, evaluator.n, model.hyb_poles.size(), triqs_xca::diagram::Kind::self_energy);
  long samples = std::min(count, 64L);
  nda::vector<long> indices(samples);
  std::vector<triqs_xca::diagram::Indices> logical;
  std::vector<triqs_xca::diagram::Diagram> diagrams;
  // Stratified samples with a fixed seed avoid aliasing repeated low-order digits.
  std::mt19937_64 random{1789};
  for (long i = 0; i < samples; ++i) {
    long first = (count / samples) * i + (count % samples) * i / samples;
    long last  = (count / samples) * (i + 1) + (count % samples) * (i + 1) / samples;
    indices(i) = std::uniform_int_distribution<long>{first, last - 1}(random);
    logical.push_back(triqs_xca::diagram::decode_indices(order, evaluator.n, model.hyb_poles.size(), indices(i), triqs_xca::diagram::Kind::self_energy));
    diagrams.push_back(triqs_xca::diagram::prepare_diagram(topology, evaluator.n, logical.back(), model.hyb_poles, triqs_xca::diagram::Kind::self_energy));
  }
  triqs_xca::block_sparse::diagram::SelfEnergyDiagram workspace(order);
  double checksum = 0;
  long paths      = 0;
  for (auto const &diagram : diagrams)
    triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, diagram, operators, support, workspace, [&](auto const &) { ++paths; });
  std::cout << "model,order,rank,sectors,max_dimension,assignment_count,samples,paths\n"
            << name << ',' << order << ',' << evaluator.r << ',' << model.ad.n_subspaces() << ',' << evaluator.Nmax << ',' << count << ',' << samples
            << ',' << paths << '\n';
  std::cout << "phase,median_us_per_assignment,min_us_per_assignment,max_us_per_assignment\n";
  auto time = [&](std::string const &phase, auto &&operation) {
    for (long i = 0; i < samples; ++i) operation(i);
    std::vector<double> timings;
    for (int trial = 0; trial < 5; ++trial) {
      auto start = Clock::now();
      for (int repeat = 0; repeat < repeats; ++repeat)
        for (long i = 0; i < samples; ++i) operation(i);
      timings.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / (repeats * samples));
    }
    std::ranges::sort(timings);
    std::cout << phase << ',' << timings[2] << ',' << timings.front() << ',' << timings.back() << '\n';
  };
  time("flat_preparation", [&](long i) {
    auto a = triqs_xca::diagram::prepare_diagram(topology, evaluator.n, model.hyb_poles, indices(i), triqs_xca::diagram::Kind::self_energy);
    checksum += a.direction_sign;
  });
  time("logical_preparation", [&](long i) {
    auto a = triqs_xca::diagram::prepare_diagram(topology, evaluator.n, logical[i], model.hyb_poles, triqs_xca::diagram::Kind::self_energy);
    checksum += a.direction_sign;
  });
  time("sign_and_kernel_binding", [&](long i) {
    auto signs   = triqs_xca::diagram::prepare_permutation_signs(diagrams[i], arc_catalog.statistics());
    auto kernels = triqs_xca::bind::tau_kernels(diagrams[i].kernel_placement, diagrams[i].pole_indices, evaluator.arc_data.kernel_values);
    checksum += std::abs(kernels.prefactor) + signs.with_fermionic_arc;
  });
  time("path_discovery", [&](long i) {
    triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, diagrams[i], operators, support, workspace, [&](auto const &path) { checksum += path.output_sector + 1; });
  });
  time("paths_and_operand_binding", [&](long i) {
    auto signs = triqs_xca::diagram::prepare_permutation_signs(diagrams[i], arc_catalog.statistics());
    triqs_xca::block_sparse::diagram::visit_self_energy_paths(G, diagrams[i], operators, support, workspace, [&](auto const &path) {
      auto bound = triqs_xca::block_sparse::bind::self_energy(G, diagrams[i], path, operators, support, evaluator.arc_data, arc_catalog.statistics(), signs);
      checksum += bs::select_diagonal_self_energy(bound);
    });
  });
  time("prepared_self_energy_call", [&](long i) {
    auto result = evaluator.compute_self_energy(G, diagrams[i]);
    checksum += nda::sum(nda::abs(result[0].data()));
  });
  time("flat_self_energy_call", [&](long i) {
    auto result = evaluator.compute_self_energy(G, topology.arcs, indices(i));
    checksum += nda::sum(nda::abs(result[0].data()));
  });
  long correlator_count = count / 2;
  time("single_particle_range_call", [&](long i) {
    long f      = indices(i) % correlator_count;
    auto result = evaluator.compute_single_ptcle_gf(bgf, topology.arcs, f, f + 1);
    checksum += nda::sum(nda::abs(result));
  });
  // Warm the public propagator conversion and range-call allocation sizes too.
  for (long i = 0; i < samples; ++i) {
    evaluator.compute_self_energy(bgf, topology.arcs, indices(i), indices(i) + 1);
    long f = indices(i) % correlator_count;
    evaluator.compute_single_ptcle_gf(bgf, topology.arcs, f, f + 1);
  }
  long rss_before = resident_kib();
  // Repeated public range calls reuse bounded selections and do not retain tasks.
  // Singleton ranges cover the same sample as the component timings.
  for (int repeat = 0; repeat < repeats * 10; ++repeat)
    for (long i = 0; i < samples; ++i) {
      auto result = evaluator.compute_self_energy(bgf, topology.arcs, indices(i), indices(i) + 1);
      checksum += nda::sum(nda::abs(result[0].data()));
      long f          = indices(i) % correlator_count;
      auto correlator = evaluator.compute_single_ptcle_gf(bgf, topology.arcs, f, f + 1);
      checksum += nda::sum(nda::abs(correlator));
    }
  std::cout << "rss_before_kib,rss_after_kib,range_calls_per_kind,checksum\n"
            << rss_before << ',' << resident_kib() << ',' << repeats * 10 * samples << ',' << checksum << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
