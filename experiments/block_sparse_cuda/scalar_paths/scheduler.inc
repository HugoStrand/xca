using ScalarWork = std::pair<Backbone, SelfEnergyPath>;
#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT
#include "../scalar_resident/resident.inc"
#endif
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
#include "../scalar_staging/buffers.inc"
#endif
#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
std::size_t scalar_replayed_lanes = 0;
#include "../scalar_cutoff/replay.inc"
#endif
bool scalar_eligible() const {
#ifdef XCA_EXPERIMENT_SCALAR_SETS
  return std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; });
#else
  return std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; })
     && std::all_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() == 1; });
#endif
}
std::size_t scalar_extra_bound() const {
  if (!scalar_eligible()) return 0;
  auto bound = checked_add(checked_multiply(12, tensor_bytes(1, r, r)), tensor_bytes(r, 16 * 512, 1));
#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT
  bound = checked_add(bound, checked_multiply(512, sizeof(scalar_resident_experiment::Weight)));
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
  bound = checked_add(bound, checked_multiply(512, sizeof(Complex) + sizeof(std::uintptr_t)));
#endif
#endif
#ifdef XCA_EXPERIMENT_SCALAR_SETS
  bound = checked_add(bound, tensor_bytes(r, 3 * 512, 1));
#endif
#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
  // General replay owns an extra copy of G and one returned scalar lane.
  bound = checked_add(bound, checked_add(tensor_bytes(r, model.dimensions().size(), 1), tensor_bytes(r, 1, 1)));
#endif
  return bound;
}
template <typename Input, typename Output>
void scalar_flush(std::vector<Input> const &G, std::vector<Output> &Sigma, std::vector<ScalarWork> &work,
                  scalar_edge_experiment::ScalarPack &packed) {
  if (work.empty()) return;
  long lanes = packed.f.extent(1), count = work.size();
  Tensor T(r, lanes, 1), zero(r, 1, 1), weights(r, lanes, 1), lane_result(r, 1, 1);
  ops->zero(zero);
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
  ScalarStaging staging(ops, lanes);
#endif
#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT
  prepare_resident_weights();
#endif
  auto apply_weights = [&](Tensor &target, auto get) {
    auto weight_at = [&](long lane, long t) {
      if constexpr (requires { get(work[lane].first, work[lane].second, t, lane); })
        return get(work[lane].first, work[lane].second, t, lane);
      else return get(work[lane].first, work[lane].second, t);
    };
#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT
    if constexpr (std::is_same_v<decltype(weight_at(0, 0)), scalar_resident_experiment::Weight>) {
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
      staging.host_weights() = scalar_resident_experiment::Weight{};
      for (long lane = 0; lane < count; ++lane) staging.host_weights(lane) = weight_at(lane, 0);
      staging.upload(staging.device_weights, staging.host_weights);
      scalar_edge_experiment::check(scalar_resident_experiment::apply(target.data(), staging.device_weights.data(), r, lanes, ops->stream()));
#else
      nda::vector<scalar_resident_experiment::Weight> descriptors(lanes);
      descriptors() = scalar_resident_experiment::Weight{};
      for (long lane = 0; lane < count; ++lane) descriptors(lane) = weight_at(lane, 0);
      nda::cuvector<scalar_resident_experiment::Weight> device(descriptors);
      ops->wait_for_default_uploads();
      scalar_edge_experiment::check(scalar_resident_experiment::apply(target.data(), device.data(), r, lanes, ops->stream()));
#endif
      ops->finish();
    } else
#endif
    {
    nda::array<Complex, 3> values(r, lanes, 1);
    values() = 1.;
    for (long lane = 0; lane < count; ++lane)
      for (long t = 0; t < r; ++t) {
        values(t, lane, 0) = weight_at(lane, t);
      }
    weights = Tensor(values);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(scalar_edge_experiment::multiply(target.data(), weights.data(), target.data(), target.size(), ops->stream()));
    ops->finish();
    }
  };
  auto apply_constants = [&](Tensor &target, auto get) {
#ifdef XCA_EXPERIMENT_SCALAR_CONSTANTS
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
    staging.host_constants() = 1.;
    for (long lane = 0; lane < count; ++lane) staging.host_constants(lane) = get(work[lane].first, work[lane].second, 0);
    staging.upload(staging.device_constants, staging.host_constants);
    scalar_edge_experiment::check(scalar_edge_experiment::multiply_by_lane(target.data(), staging.device_constants.data(), target.data(), target.size(), lanes, ops->stream()));
#else
    nda::vector<Complex> values(lanes);
    values() = 1.;
    for (long lane = 0; lane < count; ++lane) values(lane) = get(work[lane].first, work[lane].second, 0);
    nda::cuvector<Complex> constants(values);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(
       scalar_edge_experiment::multiply_by_lane(target.data(), constants.data(), target.data(), target.size(), lanes, ops->stream()));
#endif
    ops->finish();
#else
    apply_weights(target, get);
#endif
  };
  auto gather = [&](int edge) {
#ifdef XCA_EXPERIMENT_SCALAR_STAGING
    for (long lane = 0; lane < lanes; ++lane)
      staging.host_pointers(lane) = reinterpret_cast<std::uintptr_t>(lane < count ? G[work[lane].second.propagator_blocks[edge]].data() : zero.data());
    staging.upload(staging.device_pointers, staging.host_pointers);
    scalar_edge_experiment::check(scalar_edge_experiment::gather(staging.device_pointers.data(), staging.device_pointers.data(), packed.f.data(), packed.tmp.data(), r, lanes, ops->stream()));
#else
    nda::vector<std::uintptr_t> pointers(lanes);
    for (long lane = 0; lane < lanes; ++lane)
      pointers(lane) = reinterpret_cast<std::uintptr_t>(lane < count ? G[work[lane].second.propagator_blocks[edge]].data() : zero.data());
    nda::cuvector<std::uintptr_t> device(pointers);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(
       scalar_edge_experiment::gather(device.data(), device.data(), packed.f.data(), packed.tmp.data(), r, lanes, ops->stream()));
#endif
    ops->finish();
  };
  auto copy = [&](auto const &from, auto &to) { ops->copy(from, to); };
#ifdef XCA_EXPERIMENT_SCALAR_SETS
  auto zero_batch = [&](Tensor &target) { ops->zero(target); };
  auto add_batch = [&](Tensor const &from, Tensor &to) { ops->axpy(1., from, to); };
#endif
#include "recipe.inc"
  auto host = nda::to_host(T);
#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
  bool replayed = false;
  std::vector<Tensor> reference_input;
  for (long lane = 0; lane < count; ++lane) {
    double magnitude = 0;
    for (long t = 0; t < r; ++t) magnitude = std::max(magnitude, std::abs(host(t, lane, 0)));
    // Experimental uncertainty band, not a proven error bound for arbitrary paths.
    if (std::abs(magnitude - 1e-16) <= 1e-26) {
      if (reference_input.empty()) {
        reference_input.reserve(G.size());
        for (auto const &block : G) {
          reference_input.emplace_back(block.shape());
          ops->copy(block, reference_input.back());
        }
      }
      auto value = scalar_reference_value(reference_input, work[lane].first, work[lane].second);
      ++scalar_replayed_lanes;
      auto reference_host = nda::to_host(value);
      for (long t = 0; t < r; ++t) host(t, lane, 0) = reference_host(t, 0, 0);
      replayed = true;
    }
  }
  if (replayed) {
    T = Tensor(host);
    ops->wait_for_default_uploads();
  }
#endif
#ifdef XCA_EXPERIMENT_SCALAR_FANIN
  std::map<int, std::vector<long>> selected;
#endif
  for (long lane = 0; lane < count; ++lane) {
    double magnitude = 0;
    for (long t = 0; t < r; ++t) magnitude = std::max(magnitude, std::abs(host(t, lane, 0)));
    if (magnitude <= 1e-16) continue;
#ifdef XCA_EXPERIMENT_SCALAR_FANIN
    selected[work[lane].second.output_block].push_back(lane);
#else
    scalar_edge_experiment::check(cudaMemcpy2DAsync(lane_result.data(), sizeof(Complex), T.data() + lane, lanes * sizeof(Complex), sizeof(Complex), r,
                                                    cudaMemcpyDeviceToDevice, ops->stream()));
    ops->finish();
    ops->axpy(1., lane_result, Sigma[work[lane].second.output_block]);
#endif
    ++contributions;
  }
#ifdef XCA_EXPERIMENT_SCALAR_FANIN
  if (!selected.empty()) {
    long slots = selected.size(), accepted = 0;
    for (auto const &entry : selected) accepted += entry.second.size();
    nda::vector<std::uintptr_t> destinations(slots);
    nda::vector<long> offsets(slots + 1), indices(accepted);
    long slot = 0, position = 0;
    offsets(0) = 0;
    for (auto const &[block, sequence] : selected) {
      destinations(slot) = reinterpret_cast<std::uintptr_t>(Sigma[block].data());
      for (long lane : sequence) indices(position++) = lane;
      offsets(++slot) = position;
    }
    nda::cuvector<std::uintptr_t> device_destinations(destinations);
    nda::cuvector<long> device_offsets(offsets), device_indices(indices);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(scalar_edge_experiment::ordered_fanin(T.data(), device_destinations.data(), device_offsets.data(),
                                                                        device_indices.data(), r, lanes, slots, ops->stream()));
    ops->finish();
  }
#endif
  sample_memory();
  work.clear();
}

#ifdef XCA_EXPERIMENT_SCALAR_ARENA
#include "arena.inc"
#endif
