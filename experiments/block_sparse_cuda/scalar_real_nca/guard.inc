bool real_nca_prepared = false, real_nca_model = false;
double real_nca_field_max = 0, real_nca_weight_max = 0;
std::size_t real_nca_width = 0, real_nca_lanes = 0;
bool scalar_call_eligible(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, nda::array_const_view<int, 2> topology) {
  if (topology.extent(0) != 1 || output_placement != OutputPlacement::Resident) return false;
  if (!real_nca_prepared) {
    real_nca_prepared = true;
    real_nca_model = true;
    for (int field = 0; field < model.num_fields(); ++field)
      for (bool dag : {false, true})
        for (auto const &block : model.field(field, dag).blocks)
          for (auto z : block.values) {
            if (!std::isfinite(z.real()) || z.imag() != 0.) real_nca_model = false;
            real_nca_field_max = std::max(real_nca_field_max, std::abs(z.real()));
          }
    for (auto const &set : model.sets()) real_nca_width = std::max(real_nca_width, set.size());
    for (auto const *values : {&hyb->values, &hyb->values_reflect})
      for (auto z : *values) {
        if (!std::isfinite(z.real()) || z.imag() != 0.) real_nca_model = false;
        real_nca_weight_max = std::max(real_nca_weight_max, std::abs(z.real()));
      }
  }
  if (!real_nca_model) return false;
  double magnitude = 0;
  for (long b = 0; b < G.size(); ++b)
    for (auto z : G[b].data()) {
      if (z.imag() != 0.) return false;
      magnitude = std::max(magnitude, std::abs(z.real()));
    }
  // Bound every partial sum/product, including cancellation-heavy final sums.
  // Half the FP64 maximum leaves ample rounding headroom for int-sized sets.
  Backbone backbone(topology, model.num_fields(), model.num_interactions());
  long double paths = model.dimensions().size();
  paths *= model.sets().size();
  paths *= backbone.num_flat_indices(hyb->poles.size());
  // Keep the sequential-sum rounding factor well below the reserved margin.
  if (paths > std::ldexp(1.L, std::numeric_limits<double>::digits - 4)) return false;
  long double bound = magnitude;
  auto grow = [&](long double factor) {
    bound *= factor;
    return std::isfinite(bound) && bound <= std::numeric_limits<double>::max() / 2.L;
  };
  return grow(real_nca_field_max) && grow(real_nca_weight_max) && grow(real_nca_width)
      && grow(real_nca_field_max) && grow(real_nca_width) && grow(paths);
}
