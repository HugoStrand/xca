  auto vertex_batch = [&](int v) {
    apply_constants(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
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
    copy(T, packed.g);
    packed.compute(beta);
    copy(packed.diagonal, T);
  };
  gather(0);
  copy(packed.f, T);
  int special = work.front().second.special_vertex, m = work.front().first.m;
  for (int v = 1; v < special; ++v) {
    vertex_batch(v);
    edge_batch(v);
  }
  apply_constants(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
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
  apply_constants(T, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
    int field         = model.sets()[path.zero_set][0];
    bool forward      = !backbone.has_vertex_dag(0);
    auto const &block = model.field(field, forward).blocks[path.propagator_blocks[special - 1]];
    return block.values.empty() ? Complex{0.} : block.values[0];
  });
  for (int v = special + 1; v < 2 * m; ++v) {
    edge_batch(v - 1);
    vertex_batch(v);
  }
  apply_constants(T, [&](Backbone &backbone, SelfEnergyPath const &, long) { return Complex{kernel_prefactor(backbone)}; });
  apply_constants(T, [&](Backbone &backbone, SelfEnergyPath const &, long) {
    return Complex{double((backbone.m % 2 == 0 ? -1 : 1) * (backbone.get_fb(0) == 0 ? -1 : 1) * backbone.prefactor_sign)};
  });
