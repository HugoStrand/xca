#include "self_energy.hpp"
#include "triqs_xca/diagram/arc.hpp"

namespace triqs_xca::block_sparse::bind {
  namespace {
    constexpr auto all = nda::range::all;

    EndpointOperand bind_endpoint(BlockOpSymQuartet const &operators, int set, int index, int source_sector, bool adjoint,
                                  std::span<triqs_xca::Statistics const> statistics) {
      int flavour    = operators.sym_set_to_orb(set, index);
      auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
      return {op.get_block(source_sector)(index, all, all), flavour, statistics[flavour]};
    }
  } // namespace

  EndpointSumOperands endpoint_sum(triqs_xca::diagram::Diagram const &diagram, block_sparse::diagram::SelfEnergyDiagram const &prepared,
                                   BlockOpSymQuartet const &operators, OperatorSupportTable const &support, nda::array_const_view<nda::dcomplex, 3> tau_values,
                                   nda::array_const_view<nda::dcomplex, 3> tau_values_reflect,
                                   std::span<triqs_xca::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
    int special = diagram.topology.special_vertex();
    auto orientation = diagram.self_energy_endpoint_sum(statistics.size()).orientation;
    EndpointSumOperands endpoints{{}, {}, {}, orientation.reflection_sign(), signs};
    auto present = [&](int group, int index, int source, bool adjoint) {
      int flavour = operators.sym_set_to_orb(group, index);
      return (adjoint ? support.adjoint : support.plain)[flavour].single_target(source) >= 0;
    };
    for (int i = 0; i < operators.sym_set_sizes(prepared.early_symmetry_set); ++i) {
      bool adjoint = orientation.early_operator(-1).adjoint;
      if (present(prepared.early_symmetry_set, i, prepared.output_sector, adjoint))
        endpoints.early.push_back(bind_endpoint(operators, prepared.early_symmetry_set, i, prepared.output_sector, adjoint, statistics));
    }
    for (int i = 0; i < operators.sym_set_sizes(prepared.late_symmetry_set); ++i) {
      if (!present(prepared.late_symmetry_set, i, prepared.path.edge_sector(special - 1), orientation.late_operator(-1).adjoint)) continue;
      endpoints.late.push_back(bind_endpoint(operators, prepared.late_symmetry_set, i, prepared.path.edge_sector(special - 1),
                                             orientation.late_operator(-1).adjoint, statistics));
      int flavour = endpoints.late.back().coefficient_index;
      if (orientation.reflected())
        endpoints.coefficients.emplace_back(tau_values_reflect(all, all, flavour));
      else
        endpoints.coefficients.emplace_back(tau_values(all, flavour, all));
    }

    return endpoints;
  }

  SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram,
                                 block_sparse::diagram::SelfEnergyDiagram const &prepared, BlockOpSymQuartet const &operators,
                                 OperatorSupportTable const &support, nda::array_const_view<nda::dcomplex, 3> tau_values,
                                   nda::array_const_view<nda::dcomplex, 3> tau_values_reflect,
                                 std::span<triqs_xca::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
    int m    = diagram.topology.order();
    int sign = (m % 2 == 0 ? -1 : 1) * (diagram.arcs[0].direction == triqs_xca::diagram::Direction::forward ? 1 : -1) * diagram.direction_sign;
    return {backbone(propagator, diagram, operators, prepared.path), endpoint_sum(diagram, prepared, operators, support, tau_values, tau_values_reflect, statistics, signs),
            prepared.output_sector, sign};
  }
} // namespace triqs_xca::block_sparse::bind
