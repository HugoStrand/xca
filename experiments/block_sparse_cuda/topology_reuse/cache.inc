    // Private experiment: model, poles and mesh are immutable for this instance.
    static constexpr std::size_t maximum_prefix_cap = 8ULL * 1024 * 1024;
    std::size_t prefix_cap = maximum_prefix_cap, prefix_entry_cap = 4096, prefix_snapshot_cap = 32ULL * 1024 * 1024;
    std::map<std::vector<long>, Tensor> prefix_cache;
    std::vector<Complex> prefix_input;
    std::size_t prefix_bytes = 0, prefix_device_bytes = 0, prefix_hits = 0, prefix_misses = 0, prefix_skipped = 0;
    bool prefix_enabled = false;
    void clear_prefix_cache() {
      ops->finish();
      prefix_cache.clear();
      std::vector<Complex>{}.swap(prefix_input);
      prefix_bytes = prefix_device_bytes = prefix_hits = prefix_misses = prefix_skipped = 0;
      prefix_enabled = false;
    }
    void update_prefix_input(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) {
      auto bytes = function_bytes(r, model.dimensions());
      if (bytes > prefix_snapshot_cap) {
        clear_prefix_cache();
        return; // Explicit host snapshot cap; use independent device arithmetic.
      }
      std::vector<Complex> current;
      current.reserve(bytes / sizeof(Complex));
      for (long b = 0; b < G.size(); ++b)
        for (auto z : G[b].data()) current.push_back(z);
      if (current.size() != prefix_input.size() || std::memcmp(current.data(), prefix_input.data(), bytes) != 0) {
        clear_prefix_cache();
        prefix_input = std::move(current);
      }
      prefix_enabled = true;
    }
    void cached_prefix(Tensor &T, std::vector<Tensor> const &G, Backbone &backbone, SelfEnergyPath const &path,
                       int v, std::vector<long> &key) {
      int field = backbone.get_vertex_orb(v), pole = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
      bool bar = backbone.has_vertex_bar(v), dag = backbone.has_vertex_dag(v);
      auto const &block = (bar ? model.barred(field, pole, dag) : model.field(field, dag)).blocks[path.propagator_blocks[v - 1]];
      key.insert(key.end(), {10, field, bar, dag, bar ? pole : -1, path.propagator_blocks[v - 1], block.rows, block.columns});
      int sign = backbone.get_vertex_Ksign(v);
      if (sign) key.insert(key.end(), {11, sign, pole});
      int source = path.propagator_blocks[v];
      key.insert(key.end(), {20, source, model.dimensions()[source]});
      for (int x = 0; x < backbone.m - 1; ++x) {
        int weight = backbone.get_edge(v, x);
        if (weight) key.insert(key.end(), {21, weight, backbone.get_pole_ind(x)});
      }
      key.push_back(22);
      if (prefix_enabled) {
        auto hit = prefix_cache.find(key);
        if (hit != prefix_cache.end()) {
          Tensor out(hit->second.shape());
          ops->copy(hit->second, out);
          T = std::move(out);
          ++prefix_hits;
          return;
        }
      }
      ++prefix_misses;
      vertex(T, backbone, v, path);
      edge(T, G, backbone, v, path);
      auto charge = checked_add(tensor_bytes(r, T.extent(1), T.extent(2)),
                                checked_add(256, checked_multiply(key.size(), sizeof(long))));
      if (prefix_enabled && prefix_cache.size() < prefix_entry_cap && charge <= prefix_cap - prefix_bytes) {
        Tensor saved(T.shape());
        ops->copy(T, saved);
        prefix_cache.emplace(key, std::move(saved));
        prefix_bytes += charge;
        prefix_device_bytes += tensor_bytes(r, T.extent(1), T.extent(2));
        sample_memory();
      } else ++prefix_skipped;
    }
