#include "self_energy_contribution.hpp"
#include "triqs_xca/interaction.hpp"

namespace triqs_xca::block_sparse {
namespace {
constexpr auto all = nda::range::all;

EndpointOperand bind_endpoint(BlockOpSymQuartet const &operators, int set, int index, int source_sector,
                              bool adjoint, std::span<Statistics const> statistics) {
  int flavour = operators.sym_set_to_orb(set, index);
  auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
  return {op.get_block(source_sector)(index, all, all), flavour, statistics[flavour]};
}
}


SelfEnergyContribution bind_self_energy_contribution(BlockDiagOpFun const &propagator, DiagramAssignment const &assignment,
                                                     SelfEnergyPath const &path, BlockOpSymQuartet const &operators, OperatorSupports const &support,
                                                     hyb::Hybridization const &hybridization, std::span<Statistics const> statistics,
                                                     EndpointSigns signs) {
  int m = assignment.topology.order(), special = assignment.topology.special_vertex();
  InteractionLine line{assignment.directions(0) == 1};
  EndpointOperands endpoints{{}, {}, {}, line.reflection_sign(), signs};
  auto present = [&](int group, int index, int source, bool adjoint) {
    int flavour = operators.sym_set_to_orb(group, index);
    return (adjoint ? support.adjoint : support.plain)[flavour].single_target(source) >= 0;
  };
  for (int i = 0; i < operators.sym_set_sizes(path.kap_set); ++i) {
    bool adjoint = line.early_operator(-1).adjoint;
    if (present(path.kap_set, i, path.output_sector, adjoint))
      endpoints.early.push_back(bind_endpoint(operators, path.kap_set, i, path.output_sector, adjoint, statistics));
  }
  for (int i = 0; i < operators.sym_set_sizes(path.mu_set); ++i) {
    if (!present(path.mu_set, i, path.propagator_blocks(special - 1), line.late_operator(-1).adjoint)) continue;
    endpoints.late.push_back(bind_endpoint(operators, path.mu_set, i, path.propagator_blocks(special - 1),
                                           line.late_operator(-1).adjoint, statistics));
    int flavour = endpoints.late.back().coefficient_index;
    if (line.reflected()) endpoints.coefficients.emplace_back(hybridization.values_reflect(all, all, flavour));
    else endpoints.coefficients.emplace_back(hybridization.values(all, flavour, all));
  }

  int sign = (m % 2 == 0 ? -1 : 1) * (line.forward ? 1 : -1) * assignment.prefactor_sign;
  SelfEnergyContribution result{path, {}, {}, std::move(endpoints), sign};
  result.vertices.reserve(2 * m);
  result.propagators.reserve(2 * m - 1);
  for (int v = 0; v < 2 * m; ++v) {
    if (v == 0 || v == special) result.vertices.emplace_back();
    else result.vertices.push_back(bind_vertex_operator(operators, assignment, v, path.propagator_blocks(v - 1)));
  }
  for (int b : path.propagator_blocks) result.propagators.push_back(propagator.get_block(b));
  return result;
}

} // namespace triqs_xca::block_sparse
