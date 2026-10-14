#pragma once
#include "../../../test/c++/test_utils/correlator_precision.hpp"

// Independent scalar arithmetic diagnostic. Connectivity and cached double DLR
// constants are shared; this does not test traversal or DLR approximation error.
namespace scalar_path_experiment {
  namespace precision = triqs_xca::test_utils::precision;
  inline auto extended_self_energy(auto const &model, auto const &G, auto const &poles, auto const &topology, long first, long last) {
    using precision::Function;
    using precision::extended;
    auto mesh = G[0].mesh();
    precision::Dlr dlr(mesh.dlr_it());
    triqs_xca::hyb::Hybridization hyb(mesh, poles, model.coefficients());
    std::vector<Function> output(G.size(), Function(dlr.r, 1, 1));
    std::vector<bool> live(G.size());
    for (long b = 0; b < G.size(); ++b) {
      if (model.dimensions()[b] != 1) throw std::invalid_argument("Precision diagnostic requires scalar blocks");
      live[b] = nda::max_element(nda::abs(G[b].data())) >= 1e-16;
    }
    for (auto const &set : model.sets())
      if (set.size() != 1) throw std::invalid_argument("Precision diagnostic requires singleton sets");
    triqs_xca::block_sparse::cuda::for_each_self_energy_path(model, poles, topology, live, first, last,
      [&](auto &backbone, auto const &path) {
        Function value(G[path.propagator_blocks[0]].data());
        auto vertex = [&](int v) {
          int field = backbone.get_vertex_orb(v), pole = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v));
          bool dag = backbone.has_vertex_dag(v);
          auto const &op = backbone.has_vertex_bar(v) ? model.barred(field, pole, dag) : model.field(field, dag);
          value = precision::apply(op.blocks[path.propagator_blocks[v - 1]], value, false);
          int sign = backbone.get_vertex_Ksign(v);
          if (sign != 0)
            for (long t = 0; t < dlr.r; ++t) value(t, 0, 0) *= static_cast<long double>(sign > 0 ? hyb.k_it_p(t, pole) : hyb.k_it_m(t, pole));
        };
        auto edge = [&](int e) {
          Function input(G[path.propagator_blocks[e]].data());
          for (int x = 0; x < backbone.m - 1; ++x) {
            int sign = backbone.get_edge(e, x), pole = backbone.get_pole_ind(x);
            if (sign != 0)
              for (long t = 0; t < dlr.r; ++t) input(t, 0, 0) *= static_cast<long double>(sign > 0 ? hyb.k_it_p(t, pole) : hyb.k_it_m(t, pole));
          }
          value = dlr.convolve(mesh.beta(), dlr.solve(input), dlr.solve(value));
        };
        for (int v = 1; v < path.special_vertex; ++v) { vertex(v); edge(v); }
        int field = model.sets()[path.zero_set][0];
        bool forward = !backbone.has_vertex_dag(0);
        value = precision::apply(model.field(field, !forward).blocks[path.output_block], value, true);
        backbone.set_orb_inds_of_0_and_vct0(field, field);
        int parity = backbone.get_parity() * (forward ? 1 : -1);
        for (long t = 0; t < dlr.r; ++t)
          value(t, 0, 0) *= static_cast<long double>(parity) * extended(forward ? hyb.values(t, field, field) : hyb.values_reflect(t, field, field));
        value = precision::apply(model.field(field, forward).blocks[path.propagator_blocks[path.special_vertex - 1]], value, false);
        for (int v = path.special_vertex + 1; v < 2 * backbone.m; ++v) { edge(v - 1); vertex(v); }
        double prefactor = 1.;
        for (int x = 0; x < backbone.m - 1; ++x) {
          int exponent = backbone.get_prefactor_Kexp(x), pole = backbone.get_pole_ind(x);
          if (exponent != 0) prefactor *= std::pow(backbone.get_prefactor_Ksign(x) > 0 ? hyb.k_0_p(pole) : hyb.k_0_m(pole), -exponent);
        }
        int sign = (backbone.m % 2 == 0 ? -1 : 1) * (backbone.get_fb(0) == 0 ? -1 : 1) * backbone.prefactor_sign;
        long double magnitude = 0;
        for (auto &z : value.data) { z *= static_cast<long double>(prefactor); z *= sign; magnitude = std::max(magnitude, std::abs(z)); }
        if (magnitude > 1e-16)
          for (long t = 0; t < dlr.r; ++t) output[path.output_block](t, 0, 0) += value(t, 0, 0);
      });
    return output;
  }
}
