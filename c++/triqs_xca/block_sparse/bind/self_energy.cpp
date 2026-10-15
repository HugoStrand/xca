#include "self_energy.hpp"
#include "triqs_xca/diagram/arc.hpp"

namespace triqs_xca::block_sparse::bind {
  namespace {
    constexpr auto all = nda::range::all;

    EndpointOperand bind_endpoint(BlockOpSymQuartet const &operators, int set, int index, int source_sector, bool adjoint,
                                  std::span<triqs_xca::diagram::Statistics const> statistics) {
      int flavour    = operators.sym_set_to_orb(set, index);
      auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
      return {op.get_block(source_sector)(index, all, all), flavour, statistics[flavour]};
    }
  } // namespace

  EndpointSumOperands endpoint_sum(triqs_xca::diagram::Diagram const &diagram, block_sparse::diagram::SelfEnergyDiagram const &prepared,
                                   BlockOpSymQuartet const &operators, OperatorSupportTable const &support, ArcData const &arc_data,
                                   std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
    int special = diagram.topology.special_vertex();
    triqs_xca::diagram::ArcOrientation orientation{diagram.directions(0) == 1};
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
        endpoints.coefficients.emplace_back(arc_data.values_reflect(all, all, flavour));
      else
        endpoints.coefficients.emplace_back(arc_data.values(all, flavour, all));
    }

    return endpoints;
  }

  SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram,
                                 block_sparse::diagram::SelfEnergyDiagram const &prepared, BlockOpSymQuartet const &operators,
                                 OperatorSupportTable const &support, ArcData const &arc_data,
                                 std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs) {
    int m    = diagram.topology.order();
    int sign = (m % 2 == 0 ? -1 : 1) * (diagram.directions(0) == 1 ? 1 : -1) * diagram.direction_sign;
    return {backbone(propagator, diagram, operators, prepared.path), endpoint_sum(diagram, prepared, operators, support, arc_data, statistics, signs),
            prepared.output_sector, sign};
  }
} // namespace triqs_xca::block_sparse::bind
