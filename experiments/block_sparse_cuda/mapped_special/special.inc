std::size_t map_bound() const {
  std::size_t q = 0;
  for (auto const &set : model.sets())
    if (set.size() > 1) q = std::max(q, set.size());
  auto maximum = *std::max_element(model.dimensions().begin(), model.dimensions().end());
  return checked_multiply(q, checked_add(checked_multiply(maximum, sizeof(long) + sizeof(Complex)), checked_multiply(r, sizeof(Complex))));
}
void special(Tensor &T, Backbone &backbone, SelfEnergyPath const &path) {
  int s = path.special_vertex, source = path.propagator_blocks[s - 1];
  bool forward    = !backbone.has_vertex_dag(0);
  auto const &set = model.sets()[path.zero_set];
  if (set.size() <= 1) return special_reference(T, backbone, path);
  std::vector<SparseBlock const *> blocks;
  for (int field : set) blocks.push_back(&model.field(field, !forward).blocks[path.output_block]);
  auto maps = mapped_special_experiment::column_maps(blocks);
  if (!maps) return special_reference(T, backbone, path);
  if (T.shape() != std::array<long, 3>{r, path.dimensions[s], blocks.front()->rows} || maps->sources.extent(1) != path.dimensions[0])
    throw std::invalid_argument("Mapped kappa shape mismatch");
  nda::cuarray<long, 2> indices(maps->sources);
  nda::cuarray<Complex, 2> values(maps->values);
  Tensor out(r, path.dimensions[s + 1], path.dimensions[0]), mu(r, path.dimensions[s], path.dimensions[0]), term(out.shape());
  ops->zero(out);
  for (int i : set) {
    nda::array<Complex, 2> weights(r, set.size());
    for (std::size_t k = 0; k < set.size(); ++k) {
      int j = set[k];
      backbone.set_orb_inds_of_0_and_vct0(j, i);
      int sign = backbone.get_parity() * (forward ? 1 : -1);
      for (long t = 0; t < r; ++t) weights(t, k) = sign * (forward ? hyb->values(t, i, j) : hyb->values_reflect(t, j, i));
    }
    nda::cuarray<Complex, 2> device_weights(weights);
    ops->wait_for_default_uploads();
    if (mapped_special_experiment::weighted_mu(T.data(), indices.data(), values.data(), device_weights.data(), mu.data(), r, T.extent(1), T.extent(2),
                                               mu.extent(2), set.size(), ops->stream())
        != cudaSuccess)
      throw std::runtime_error("Mapped kappa kernel launch failed");
    ops->finish();
    device_operator(model.field(i, forward).blocks[source]).apply_left(mu, term);
    ops->axpy(1., term, out);
    sample_memory();
  }
  T = std::move(out);
}
