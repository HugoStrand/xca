#include <triqs_xca/block_sparse/self_energy_paths.hpp>
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
#include <triqs_xca/action.hpp>

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
  nda::array<int, 2> topology(order, 2);
  for (int a = 0; a < order; ++a) topology(a, nda::range::all) = nda::vector<int>{a, a + order};
  triqs_xca::diagram::topology::validate(topology);
  long count   = triqs_xca::diagram::count_diagrams(order, evaluator.action.n_flavours(), model.hyb_poles.size(), triqs_xca::diagram::Kind::self_energy);
  long samples = std::min(count, 64L);
  nda::vector<long> indices(samples);
  std::vector<triqs_xca::diagram::Arcs> logical;
  std::vector<triqs_xca::diagram::SelfEnergyBackbone> backbones;
  std::vector<triqs_xca::diagram::CorrelatorBackbone> correlator_backbones;
  long correlator_count = triqs_xca::diagram::count_diagrams(order, evaluator.action.n_flavours(), model.hyb_poles.size(), triqs_xca::diagram::Kind::correlator);
  auto [mu, kap] = tu::make_correlator_ops(evaluator.action.operators, evaluator.action.n_hyb);
  // Stratified samples with a fixed seed avoid aliasing repeated low-order digits.
  std::mt19937_64 random{1789};
  for (long i = 0; i < samples; ++i) {
    long first = (count / samples) * i + (count % samples) * i / samples;
    long last  = (count / samples) * (i + 1) + (count % samples) * (i + 1) / samples;
    indices(i) = std::uniform_int_distribution<long>{first, last - 1}(random);
    logical.push_back(triqs_xca::diagram::decode_arcs(order, evaluator.action, indices(i), triqs_xca::diagram::Kind::self_energy));
    backbones.push_back(triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, logical.back()));
    correlator_backbones.push_back(triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(
      topology, evaluator.action, indices(i) % correlator_count));
  }
  double checksum = 0;
  long paths      = 0;
  for (auto const &backbone : backbones)
    triqs_xca::block_sparse::visit_self_energy_paths(G, backbone, operators, support, [&](auto const &) { ++paths; });
  std::cout << "model,order,rank,sectors,max_dimension,assignment_count,samples,paths\n"
            << name << ',' << order << ',' << evaluator.action.tau_mesh.size() << ',' << model.ad.n_subspaces() << ',' << nda::max_element(evaluator.action.block_dims) << ',' << count << ',' << samples
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
    auto a = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(
      topology, triqs_xca::diagram::decode_arcs(order, evaluator.action, indices(i), triqs_xca::diagram::Kind::self_energy));
    checksum += a.direction_sign;
  });
  time("logical_preparation", [&](long i) {
    auto a = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, logical[i]);
    checksum += a.direction_sign;
  });
  time("kernel_prefactor", [&](long i) {
    auto signs   = backbones[i].endpoints.signs;
    double prefactor = tau_kernel_prefactor(backbones[i].kernel_prefactor, evaluator.action.kernel);
    checksum += std::abs(prefactor) + signs.with_fermionic_arc;
  });
  time("path_discovery", [&](long i) {
    triqs_xca::block_sparse::visit_self_energy_paths(G, backbones[i], operators, support, [&](auto const &path) { checksum += path.sector(0) + 1; });
  });
  time("prepared_self_energy_call", [&](long i) {
    auto result = evaluator.compute_self_energy(G, backbones[i]);
    checksum += nda::sum(nda::abs(result.get_block(0)));
  });
  time("prepare_and_self_energy_call", [&](long i) {
    auto backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, evaluator.action, indices(i));
    auto result = evaluator.compute_self_energy(G, backbone);
    checksum += nda::sum(nda::abs(result.get_block(0)));
  });
  time("prepared_single_particle_call", [&](long i) {
    auto result = evaluator.compute_one_time_correlator(G, correlator_backbones[i], mu, kap);
    checksum += nda::sum(nda::abs(result));
  });
  // Warm both local prepared call paths before sampling resident memory.
  for (long i = 0; i < samples; ++i) {
    evaluator.compute_self_energy(G, backbones[i]);
    evaluator.compute_one_time_correlator(G, correlator_backbones[i], mu, kap);
  }
  long rss_before = resident_kib();
  // The same bounded backbone sample is reused; no task history is retained.
  for (int repeat = 0; repeat < repeats * 10; ++repeat)
    for (long i = 0; i < samples; ++i) {
      auto result = evaluator.compute_self_energy(G, backbones[i]);
      checksum += nda::sum(nda::abs(result.get_block(0)));
      auto correlator = evaluator.compute_one_time_correlator(G, correlator_backbones[i], mu, kap);
      checksum += nda::sum(nda::abs(correlator));
    }
  std::cout << "rss_before_kib,rss_after_kib,prepared_calls_per_kind,checksum\n"
            << rss_before << ',' << resident_kib() << ',' << repeats * 10 * samples << ',' << checksum << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
