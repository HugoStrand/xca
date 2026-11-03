#include "test_utils/evaluators.hpp"
#include <triqs_xca/block_sparse/backbone_paths.hpp>
#include <triqs_xca/block_sparse/correlator_paths.hpp>
#include <triqs_xca/block_sparse/self_energy_paths.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {
  namespace xca     = triqs_xca;
  namespace bs      = xca::block_sparse;
  namespace diagram = xca::diagram;
  namespace tu      = xca::test_utils;
  using nda::dcomplex;
  using Clock                      = std::chrono::steady_clock;
  constexpr auto all               = nda::range::all;
  constexpr long requested_samples = 8, candidate_limit = 512;
  constexpr int trials = 3;
  double checksum      = 0.;

  tu::EvalModel model(std::string const &name, bool high_rank) {
    auto result = [&] {
      if (name != "density4") return tu::build_eval_model(name);
      triqs::hilbert_space::fundamental_operator_set fops;
      triqs::operators::many_body_operator_complex H;
      for (int i = 0; i < 4; ++i) {
        fops.insert("a", i);
        H += (0.13 + 0.17 * i) * triqs::operators::n("a", i);
        for (int j = 0; j < i; ++j) H += 0.6 * triqs::operators::n("a", i) * triqs::operators::n("a", j);
      }
      return tu::make_eval_model(name, tu::ad_t(H, fops));
    }();
    if (high_rank) {
      constexpr double beta = 2., lambda = 1e5, eps = 1e-12;
      result.mesh = tu::mesh_t(beta, triqs::mesh::Fermion, lambda / beta, eps, false);
      auto ops    = result.mesh.dlr_it();
      auto atomic = tu::ad_to_atom_prop(result.ad, beta, ops);
      std::vector<triqs::gfs::gf<tu::mesh_t>> blocks;
      for (int b = 0; b < result.ad.n_subspaces(); ++b) blocks.emplace_back(result.mesh, atomic.get_block(b));
      result.G = tu::dressed_propagator(result.ad, result.mesh, tu::bgf_t(blocks));
    }
    return result;
  }

  nda::vector<dcomplex> flatten_result(bs::BlockDiagOpFun const &result) {
    long entries = 0;
    for (auto const &block : result.get_blocks()) entries += block.size();
    nda::vector<dcomplex> flat(entries);
    long offset = 0;
    for (auto const &block : result.get_blocks())
      for (auto value : block) flat(offset++) = value;
    return flat;
  }

  nda::vector<dcomplex> flatten_result(nda::array<dcomplex, 3> const &result) {
    nda::vector<dcomplex> flat(result.size());
    std::copy(result.begin(), result.end(), flat.begin());
    return flat;
  }

  void consume(bs::BlockDiagOpFun const &result) { checksum += result.get_block(0)(0, 0, 0).real(); }
  void consume(nda::array<dcomplex, 3> const &result) { checksum += result(0, 0, 0).real(); }

  double magnitude(nda::vector_const_view<dcomplex> result) {
    for (auto value : result)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) throw std::runtime_error("nonfinite benchmark output");
    return nda::max_element(nda::abs(result));
  }

  double compare(nda::vector_const_view<dcomplex> actual, nda::vector_const_view<dcomplex> expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("baseline output shape changed");
    double norm = magnitude(expected), error = magnitude(nda::make_regular(actual - expected));
    // No absolute floor: zero-path references must stay exactly zero.
    if (error > 2e-10 * norm) throw std::runtime_error("elementwise comparison to recorded baseline failed");
    return norm == 0 ? 0. : error / norm;
  }

  struct Paths {
    long completed = 0, segments = 0;
  };

  Paths paths(bs::BlockDiagOpFun const &G, diagram::SelfEnergyBackbone const &backbone, xca::Action<bs::BlockOpSymQuartet> const &action,
              bs::OperatorSupportTable const &support, std::vector<bs::BlockOp> const &, std::vector<bs::BlockOp> const &) {
    Paths count;
    bs::visit_self_energy_paths(G, backbone, action.operators, support, [&](auto const &path) {
      if (!path.zero_operators.empty() && !path.special_operators.empty() && path.sector(0) == path.sector(path.range.vertex_end)) ++count.completed;
    });
    return count;
  }

  Paths paths(bs::BlockDiagOpFun const &G, diagram::CorrelatorBackbone const &backbone, xca::Action<bs::BlockOpSymQuartet> const &action,
              bs::OperatorSupportTable const &support, std::vector<bs::BlockOp> const &mu, std::vector<bs::BlockOp> const &kap) {
    Paths count;
    std::vector<bs::diagram::CorrelatorBackbonePath> before, after;
    auto ranges = backbone.endpoint_ranges();
    bs::visit_backbone_paths(G, backbone, action.operators, support, ranges.before_special, [&](auto const &path) { before.push_back(path); });
    bs::visit_backbone_paths(G, backbone, action.operators, support, ranges.after_special, [&](auto const &path) { after.push_back(path); });
    count.segments = before.size() + after.size();
    bs::visit_correlator_paths(G, before, after, mu, kap, [&](auto const &path, int, int) {
      if (!path.zero_operators.empty() && !path.special_operators.empty()) ++count.completed;
    });
    return count;
  }

  template <diagram::Kind K, typename Run>
  void benchmark(std::string const &label, int order, bool record, h5::group archive, std::ofstream &manifest, bs::DiagramEvaluator &evaluator,
                 bs::BlockDiagOpFun const &G, bs::OperatorSupportTable const &support, std::vector<bs::BlockOp> const &mu,
                 std::vector<bs::BlockOp> const &kap, Run run) {
    nda::array<int, 2> topology(order, 2);
    for (int a = 0; a < order; ++a) {
      topology(a, 0) = a;
      topology(a, 1) = a + order;
    }
    long count = diagram::count_diagrams(order, evaluator.action.n_flavours(), evaluator.action.poles.size(), K);
    std::array<std::vector<long>, 2> selected;
    std::array<std::vector<nda::vector<dcomplex>>, 2> references;
    long attempted = 0;
    if (record) {
      std::mt19937_64 random{1789 + 97 * static_cast<unsigned long>(order) + static_cast<unsigned long>(K)};
      std::unordered_set<long> visited;
      for (long draw = 0; draw < 4 * candidate_limit && attempted < std::min(count, candidate_limit); ++draw) {
        long index = std::uniform_int_distribution<long>{0, count - 1}(random);
        if (!visited.insert(index).second) continue;
        ++attempted;
        auto backbone   = diagram::prepare_backbone<K>(topology, evaluator.action, index);
        auto discovered = paths(G, backbone, evaluator.action, support, mu, kap);
        int group       = discovered.completed == 0 ? 1 : 0;
        if (selected[group].size() == requested_samples) continue;
        auto output = flatten_result(run(backbone));
        double norm = magnitude(output);
        if (group == 0 && norm <= 1e-24) continue;
        if (group == 1 && norm != 0.) throw std::runtime_error("zero-completion fixture has nonzero result");
        selected[group].push_back(index);
        references[group].push_back(std::move(output));
        if (selected[0].size() == requested_samples && selected[1].size() == requested_samples) break;
      }
      if (selected[0].empty()) throw std::runtime_error("no productive samples for " + label);
    }
    for (int group = 0; group < 2; ++group) {
      std::string key = label + (group == 0 ? "_productive" : "_zero_path");
      if (record && selected[group].empty()) {
        std::cerr << "No zero-completion sample in bounded search: " << label << '\n';
        continue;
      }
      if (!record && !archive.has_subgroup(key)) {
        if (group == 0) throw std::runtime_error("missing productive baseline case: " + key);
        continue; // A bounded search need not find a zero-path sample.
      }
      auto store = record ? archive.create_group(key) : archive.open_group(key);
      nda::vector<long> indices;
      nda::array<dcomplex, 2> expected;
      if (record) {
        indices.resize(selected[group].size());
        std::copy(selected[group].begin(), selected[group].end(), indices.begin());
        expected.resize(indices.size(), references[group].front().size());
        for (long sample = 0; sample < indices.size(); ++sample) expected(sample, all) = references[group][sample];
        h5::write(store, "indices", indices);
        h5::write(store, "outputs", expected);
        h5::write(store, "block_dims", evaluator.action.block_dims);
      } else {
        h5::read(store, "indices", indices);
        h5::read(store, "outputs", expected);
        if (expected.extent(0) != indices.size() || indices.empty()) throw std::runtime_error("invalid baseline sample table");
        nda::vector<long> dimensions;
        h5::read(store, "block_dims", dimensions);
        if (dimensions != evaluator.action.block_dims) throw std::runtime_error("baseline sector layout changed");
      }
      std::vector<diagram::Backbone<K>> backbones;
      long complete_paths = 0, segments = 0;
      double max_error = 0., output_max = 0.;
      for (long sample = 0; sample < indices.size(); ++sample) {
        backbones.push_back(diagram::prepare_backbone<K>(topology, evaluator.action, indices(sample)));
        auto discovered = paths(G, backbones.back(), evaluator.action, support, mu, kap);
        complete_paths += discovered.completed;
        segments += discovered.segments;
        auto output = flatten_result(run(backbones.back()));
        max_error   = std::max(max_error, compare(output, expected(sample, all)));
        output_max  = std::max(output_max, magnitude(expected(sample, all)));
        if (record)
          manifest << key << ',' << indices(sample) << ',' << discovered.completed << ',' << discovered.segments << ',' << magnitude(output) << '\n';
      }
      auto pass = [&] {
        for (auto const &backbone : backbones) {
          auto output = run(backbone);
          consume(output);
        }
      };
      auto start = Clock::now();
      pass();
      double warmup = std::chrono::duration<double>(Clock::now() - start).count();
      int repeats   = std::clamp(static_cast<int>(0.003 / std::max(warmup, 1e-9)), 1, 5);
      std::vector<double> timings;
      for (int trial = 0; trial < trials; ++trial) {
        start = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) pass();
        timings.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / (repeats * indices.size()));
      }
      for (long sample = 0; sample < indices.size(); ++sample)
        max_error = std::max(max_error, compare(flatten_result(run(backbones[sample])), expected(sample, all)));
      std::ranges::sort(timings);
      std::cout << key << ',' << evaluator.action.tau_mesh.size() << ',' << order << ',' << evaluator.action.block_dims.size() << ','
                << nda::max_element(evaluator.action.block_dims) << ',' << count << ',' << indices.size() << ',' << attempted << ',' << complete_paths
                << ',' << segments << ',' << timings[trials / 2] << ',' << timings.front() << ',' << timings.back() << ',' << repeats << ',' << trials
                << ',' << max_error << ',' << output_max << '\n';
      std::cout.flush();
    }
  }
} // namespace

int main(int argc, char **argv) try {
  if (argc < 3 || argc > 4 || (std::string(argv[1]) != "record" && std::string(argv[1]) != "compare"))
    throw std::invalid_argument("usage: block_sparse_convolution {record|compare} BASELINE_PREFIX [case_substring]");
  bool record        = std::string(argv[1]) == "record";
  std::string prefix = argv[2], filter = argc == 4 ? argv[3] : "";
  // Exclusive create prevents accidentally replacing the pre-optimization reference.
  h5::file file(prefix + ".h5", record ? 'e' : 'r');
  h5::group archive(file);
  std::ofstream manifest;
  if (record) {
    manifest.open(prefix + ".indices.csv");
    if (!manifest) throw std::runtime_error("cannot write baseline ordinal manifest");
    manifest << std::setprecision(17) << "case,index,completed_paths,segments,output_max\n";
  }
  std::cout
     << std::setprecision(17)
     << "case,rank,order,sectors,max_dimension,total_backbones,samples,selection_attempts,completed_paths,segments,median_us_per_backbone,min_us_per_backbone,max_us_per_backbone,repeats,trials,relative_max_error,output_max\n";
  long cases = 0;
  for (std::string const name : {"density4", "unequal_sets_nA0", "n_partition_spin_xyz"})
    for (bool high_rank : {false, true}) {
      auto m = model(name, high_rank);
      bs::DiagramEvaluator evaluator(m.hyb_poles, m.hyb_coeffs, m.mesh, m.ad, m.dynint_ops, m.dynint_coeffs);
      bs::BlockDiagOpFun G(tu::subspace_blocks(m.G, m.ad, m.mesh));
      bs::OperatorSupportTable support(evaluator.action.operators);
      auto [mu, kap] = tu::make_correlator_ops(evaluator.action.operators, evaluator.action.n_hyb);
      for (int order : {2, 3, 5}) {
        auto base  = name + "_r" + std::to_string(m.mesh.size()) + "_order" + std::to_string(order);
        auto label = base + "_self_energy";
        if (filter.empty() || label.find(filter) != std::string::npos) {
          ++cases;
          benchmark<diagram::Kind::self_energy>(label, order, record, archive, manifest, evaluator, G, support, mu, kap,
                                                [&](auto const &backbone) { return evaluator.compute_self_energy(G, backbone); });
        }
        label = base + "_correlator";
        if (filter.empty() || label.find(filter) != std::string::npos) {
          ++cases;
          benchmark<diagram::Kind::correlator>(label, order, record, archive, manifest, evaluator, G, support, mu, kap,
                                               [&](auto const &backbone) { return evaluator.compute_one_time_correlator(G, backbone, mu, kap); });
        }
      }
    }
  if (cases == 0) throw std::runtime_error("case filter matched no cases");
  std::cerr << "checksum=" << std::setprecision(17) << checksum << '\n';
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
