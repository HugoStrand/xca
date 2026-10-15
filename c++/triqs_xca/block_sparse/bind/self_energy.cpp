#include "self_energy.hpp"
#include "triqs_xca/diagram/arc.hpp"

namespace triqs_xca::block_sparse::bind {
namespace {
constexpr auto all = nda::range::all;

EndpointOperand bind_endpoint(BlockOpSymQuartet const &operators, int set, int index, int source_sector,
                              bool adjoint, std::span<triqs_xca::diagram::Statistics const> statistics) {
  int flavour = operators.sym_set_to_orb(set, index);
  auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
  return {op.get_block(source_sector)(index, all, all), flavour, statistics[flavour]};
}
}


EndpointSumOperands endpoint_sum(triqs_xca::diagram::Diagram const &assignment,
    block_sparse::diagram::SelfEnergyDiagram const &path, BlockOpSymQuartet const &operators,
    OperatorSupports const &support, ArcData const &hybridization,
    std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
  int special = assignment.topology.special_vertex();
  triqs_xca::diagram::ArcOrientation line{assignment.directions(0) == 1};
  EndpointSumOperands endpoints{{}, {}, {}, line.reflection_sign(), signs};
  auto present = [&](int group, int index, int source, bool adjoint) {
    int flavour = operators.sym_set_to_orb(group, index);
    return (adjoint ? support.adjoint : support.plain)[flavour].single_target(source) >= 0;
  };
  for (int i = 0; i < operators.sym_set_sizes(path.early_symmetry_set); ++i) {
    bool adjoint = line.early_operator(-1).adjoint;
    if (present(path.early_symmetry_set, i, path.output_sector, adjoint))
      endpoints.early.push_back(bind_endpoint(operators, path.early_symmetry_set, i, path.output_sector, adjoint, statistics));
  }
  for (int i = 0; i < operators.sym_set_sizes(path.late_symmetry_set); ++i) {
    if (!present(path.late_symmetry_set, i, path.path.edge_sector(special - 1), line.late_operator(-1).adjoint)) continue;
    endpoints.late.push_back(bind_endpoint(operators, path.late_symmetry_set, i, path.path.edge_sector(special - 1),
                                           line.late_operator(-1).adjoint, statistics));
    int flavour = endpoints.late.back().coefficient_index;
    if (line.reflected()) endpoints.coefficients.emplace_back(hybridization.values_reflect(all, all, flavour));
    else endpoints.coefficients.emplace_back(hybridization.values(all, flavour, all));
  }

  return endpoints;
}

SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
    block_sparse::diagram::SelfEnergyDiagram const &path, BlockOpSymQuartet const &operators,
    OperatorSupports const &support, ArcData const &hybridization,
    std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
  int m = assignment.topology.order();
  int sign = (m % 2 == 0 ? -1 : 1) * (assignment.directions(0) == 1 ? 1 : -1) * assignment.direction_sign;
  return {backbone(propagator, assignment, operators, path.path),
          endpoint_sum(assignment, path, operators, support, hybridization, statistics, signs),
          path.output_sector, sign};
}
} // namespace triqs_xca::block_sparse::bind
