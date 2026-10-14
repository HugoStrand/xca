triqs::gfs::block_gf<triqs::mesh::dlr_imtime> scalar_arena_compute(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,
                                                                   nda::array_const_view<int, 2> topology, std::vector<bool> const &live, long first,
                                                                   long last) {
  std::size_t total = 0;
  if (cudaMemGetInfo(&baseline_free, &total) != cudaSuccess) throw std::runtime_error("Scalar arena memory query failed");
  long blocks     = G.size();
  auto host_input = nda::zeros<Complex>(blocks, r, 1);
  for (long b = 0; b < blocks; ++b)
    if (live[b])
      for (long t = 0; t < r; ++t) host_input(b, t, 0) = G[b].data()(t, 0, 0);
  Tensor input_storage(host_input), output_storage(blocks, r, 1);
  ops->zero(output_storage);
  constexpr auto all = nda::range::all;
  std::vector<TensorConstView> input;
  std::vector<TensorView> output;
  input.reserve(blocks);
  output.reserve(blocks);
  for (long b = 0; b < blocks; ++b) {
    input.emplace_back(nda::reshape(input_storage(b, all, all), r, 1, 1));
    output.emplace_back(nda::reshape(output_storage(b, all, all), r, 1, 1));
  }
  std::unique_ptr<scalar_edge_experiment::ScalarPack> packed;
  std::vector<ScalarWork> work;
  work.reserve(512);
  auto flush = [&] {
    if (work.empty()) return;
    if (!packed) packed = std::make_unique<scalar_edge_experiment::ScalarPack>(ops, mesh.dlr_it(), 512);
    scalar_flush(input, output, work, *packed);
  };
  counts = for_each_self_energy_path(model, hyb->poles, topology, live, first, last, [&](Backbone &backbone, SelfEnergyPath const &path) {
    if (!work.empty() && work.front().second.special_vertex != path.special_vertex) flush();
    work.emplace_back(backbone, path);
    if (work.size() == 512) flush();
  });
  flush();
  sample_memory();
  auto host = nda::to_host(output_storage);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> result;
  for (long b = 0; b < blocks; ++b) result.emplace_back(mesh, nda::reshape(host(b, all, all), r, 1, 1));
  return {result};
}
