using ScalarWork = std::pair<Backbone, SelfEnergyPath>;
bool scalar_eligible() const {
  return std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; })
     && std::all_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() == 1; });
}
void scalar_flush(std::vector<Tensor> const &G, std::vector<Tensor> &Sigma, std::vector<ScalarWork> &work,
                  scalar_edge_experiment::ScalarPack &packed) {
  if (work.empty()) return;
  long lanes = packed.f.extent(1), count = work.size();
  Tensor T(r, lanes, 1), zero(r, 1, 1), weights(r, lanes, 1), lane_result(r, 1, 1);
  ops->zero(zero);
  auto apply_weights = [&](Tensor &target, auto get) {
    nda::array<Complex, 3> values(r, lanes, 1);
    values() = 1.;
    for (long lane = 0; lane < count; ++lane)
      for (long t = 0; t < r; ++t) values(t, lane, 0) = get(work[lane].first, work[lane].second, t);
    weights = Tensor(values);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(scalar_edge_experiment::multiply(target.data(), weights.data(), target.data(), target.size(), ops->stream()));
    ops->finish();
  };
  auto gather = [&](int edge) {
    nda::vector<std::uintptr_t> pointers(lanes);
    for (long lane = 0; lane < lanes; ++lane)
      pointers(lane) = reinterpret_cast<std::uintptr_t>(lane < count ? G[work[lane].second.propagator_blocks[edge]].data() : zero.data());
    nda::cuvector<std::uintptr_t> device(pointers);
    ops->wait_for_default_uploads();
    scalar_edge_experiment::check(
       scalar_edge_experiment::gather(device.data(), device.data(), packed.f.data(), packed.tmp.data(), r, lanes, ops->stream()));
    ops->finish();
  };
  auto vertex_batch = [&](int v) {
    apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
      int field = backbone.get_vertex_orb(v), p = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
      bool dag          = backbone.has_vertex_dag(v);
      auto const &op    = backbone.has_vertex_bar(v) ? model.barred(field, p, dag) : model.field(field, dag);
      auto const &block = op.blocks[path.propagator_blocks[v - 1]];
      return block.values.empty() ? Complex{0.} : block.values[0];
    });
    apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &, long t) {
      int sign = backbone.get_vertex_Ksign(v), p = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
      return sign == 0 ? Complex{1.} : Complex{sign > 0 ? hyb->k_it_p(t, p) : hyb->k_it_m(t, p)};
    });
  };
  auto edge_batch = [&](int e) {
    gather(e);
    for (int x = 0; x < work.front().first.m - 1; ++x)
      apply_weights(packed.f, [&](Backbone &backbone, SelfEnergyPath const &, long t) {
        int sign = backbone.get_edge(e, x), p = backbone.get_pole_ind(x);
        return sign == 0 ? Complex{1.} : Complex{sign > 0 ? hyb->k_it_p(t, p) : hyb->k_it_m(t, p)};
      });
    ops->copy(T, packed.g);
    packed.compute(beta);
    ops->copy(packed.diagonal, T);
  };
  gather(0);
  ops->copy(packed.f, T);
  int special = work.front().second.special_vertex, m = work.front().first.m;
  for (int v = 1; v < special; ++v) {
    vertex_batch(v);
    edge_batch(v);
  }
  apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
    int field         = model.sets()[path.zero_set][0];
    bool forward      = !backbone.has_vertex_dag(0);
    auto const &block = model.field(field, !forward).blocks[path.output_block];
    backbone.set_orb_inds_of_0_and_vct0(field, field);
    return block.values.empty() ? Complex{0.} : block.values[0];
  });
  apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &path, long t) {
    int field    = model.sets()[path.zero_set][0];
    bool forward = !backbone.has_vertex_dag(0);
    int sign     = backbone.get_parity() * (forward ? 1 : -1);
    return Complex{static_cast<double>(sign)} * (forward ? hyb->values(t, field, field) : hyb->values_reflect(t, field, field));
  });
  apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
    int field         = model.sets()[path.zero_set][0];
    bool forward      = !backbone.has_vertex_dag(0);
    auto const &block = model.field(field, forward).blocks[path.propagator_blocks[special - 1]];
    return block.values.empty() ? Complex{0.} : block.values[0];
  });
  for (int v = special + 1; v < 2 * m; ++v) {
    edge_batch(v - 1);
    vertex_batch(v);
  }
  apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &, long) { return Complex{kernel_prefactor(backbone)}; });
  apply_weights(T, [&](Backbone &backbone, SelfEnergyPath const &, long) {
    return Complex{double((backbone.m % 2 == 0 ? -1 : 1) * (backbone.get_fb(0) == 0 ? -1 : 1) * backbone.prefactor_sign)};
  });
  auto host = nda::to_host(T);
  for (long lane = 0; lane < count; ++lane) {
    double magnitude = 0;
    for (long t = 0; t < r; ++t) magnitude = std::max(magnitude, std::abs(host(t, lane, 0)));
    if (magnitude <= 1e-16) continue;
    scalar_edge_experiment::check(cudaMemcpy2DAsync(lane_result.data(), sizeof(Complex), T.data() + lane, lanes * sizeof(Complex), sizeof(Complex), r,
                                                    cudaMemcpyDeviceToDevice, ops->stream()));
    ops->finish();
    ops->axpy(1., lane_result, Sigma[work[lane].second.output_block]);
    ++contributions;
  }
  sample_memory();
  work.clear();
}
