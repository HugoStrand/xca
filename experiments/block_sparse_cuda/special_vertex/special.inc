    std::size_t kappa_slots() const {
      std::size_t maximum = 0;
      for (auto const &set : model.sets()) if (set.size() > 1) maximum = std::max(maximum, set.size());
      return std::min(maximum, std::size_t(XCA_EXPERIMENT_KAPPA_SLOTS));
    }
    void special(Tensor &T, Backbone &backbone, SelfEnergyPath const &path) {
      int s = path.special_vertex, source = path.propagator_blocks[s - 1];
      bool forward = !backbone.has_vertex_dag(0);
      auto const &set = model.sets()[path.zero_set];
      Tensor out(r, path.dimensions[s + 1], path.dimensions[0]);
      ops->zero(out);
      Tensor kap(r, path.dimensions[s], path.dimensions[0]), weighted(kap.shape()), mu(kap.shape()), term(out.shape());
      // Retain only a bounded prefix of kappa products for this immutable T.
      // Mu/kappa accumulation order remains identical to the reference.
      std::size_t slots = set.size() > 1 ? std::min(set.size(), kappa_slots()) : 0;
      std::vector<Tensor> products;
      products.reserve(slots);
      for (std::size_t j = 0; j < slots; ++j) {
        products.emplace_back(kap.shape());
        device_operator(model.field(set[j], !forward).blocks[path.output_block]).apply_right(T, products.back());
        sample_memory();
      }
      for (int i : set) {
        ops->zero(mu);
        for (std::size_t index = 0; index < set.size(); ++index) {
          int j = set[index];
          if (index >= slots) device_operator(model.field(j, !forward).blocks[path.output_block]).apply_right(T, kap);
          TensorConstView product = index < slots ? TensorConstView(products[index]) : TensorConstView(kap);
          backbone.set_orb_inds_of_0_and_vct0(j, i);
          int sign = backbone.get_parity() * (forward ? 1 : -1);
          nda::vector<Complex> weights(r);
          for (long t = 0; t < r; ++t) weights(t) = sign * (forward ? hyb->values(t, i, j) : hyb->values_reflect(t, j, i));
          nda::cuvector<Complex> device_weights(weights);
          ops->weight_rows(product, device_weights, weighted);
          ops->axpy(1., weighted, mu);
          sample_memory();
        }
        device_operator(model.field(i, forward).blocks[source]).apply_left(mu, term);
        ops->axpy(1., term, out);
        sample_memory();
      }
      T = std::move(out);
    }
