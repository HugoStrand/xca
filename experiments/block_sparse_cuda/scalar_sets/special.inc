      // Preserve each lane's mu-then-kappa sum order. Missing members in shorter
      // sets contribute zero, with no cross-lane reduction or reassociation.
      Tensor out(r, lanes, 1), mu(r, lanes, 1), kap(r, lanes, 1);
      ops->zero(out);
      for (std::size_t i = 0; i < width; ++i) {
        ops->zero(mu);
        for (std::size_t j = 0; j < width; ++j) {
          copy(T, kap);
          apply_constants(kap, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
            auto const &set = model.sets()[path.zero_set];
            if (i >= set.size() || j >= set.size()) return Complex{0.};
            bool forward = !backbone.has_vertex_dag(0);
            auto const &block = model.field(set[j], !forward).blocks[path.output_block];
            return block.values.empty() ? Complex{0.} : block.values[0];
          });
          apply_weights(kap, [&](Backbone &backbone, SelfEnergyPath const &path, long t) {
            auto const &set = model.sets()[path.zero_set];
            if (i >= set.size() || j >= set.size()) return Complex{0.};
            bool forward = !backbone.has_vertex_dag(0);
            backbone.set_orb_inds_of_0_and_vct0(set[j], set[i]);
            int sign = backbone.get_parity() * (forward ? 1 : -1);
            return Complex{static_cast<double>(sign)} * (forward ? hyb->values(t, set[i], set[j]) : hyb->values_reflect(t, set[j], set[i]));
          });
          ops->axpy(1., kap, mu);
        }
        apply_constants(mu, [&](Backbone &backbone, SelfEnergyPath const &path, long) {
          auto const &set = model.sets()[path.zero_set];
          if (i >= set.size()) return Complex{0.};
          bool forward = !backbone.has_vertex_dag(0);
          auto const &block = model.field(set[i], forward).blocks[path.propagator_blocks[special - 1]];
          return block.values.empty() ? Complex{0.} : block.values[0];
        });
        ops->axpy(1., mu, out);
      }
      copy(out, T);
