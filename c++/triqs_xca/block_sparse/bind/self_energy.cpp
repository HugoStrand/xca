#include "self_energy.hpp"
#include "triqs_xca/action.hpp"

namespace triqs_xca::block_sparse::bind {
  namespace {
    constexpr auto all = nda::range::all;

    EndpointOperand bind_endpoint(BlockOpSymQuartet const &operators, int flavour, int source_sector, bool adjoint,
                                  std::span<triqs_xca::Statistics const> statistics) {
      int set = operators.sym_set_labels(flavour), index = operators.sym_set_inds(flavour);
      auto const &op = adjoint ? operators.F_dags[set] : operators.Fs[set];
      return {op.get_block(source_sector)(index, all, all), flavour, statistics[flavour]};
    }
  } // namespace

  EndpointSumOperands endpoint_sum(block_sparse::diagram::BackbonePath const &path, BlockOpSymQuartet const &operators,
                                   nda::array_const_view<nda::dcomplex, 3> tau_values, nda::array_const_view<nda::dcomplex, 3> tau_values_reflect,
                                   std::span<triqs_xca::Statistics const> statistics) {
    auto const &backbone = path.backbone;
    backbone.require_self_energy();
    if (path.range != backbone.range()) throw std::invalid_argument("endpoint sum requires a complete backbone path");
    int special = backbone.endpoints.special_vertex;
    OperatorPairOrientation orientation{backbone.endpoints.direction == triqs_xca::diagram::Direction::forward};
    EndpointSumOperands endpoints{{}, {}, {}, orientation.reflection_sign(), backbone.endpoints.signs};
    for (int flavour : path.zero_operators)
      endpoints.early.push_back(bind_endpoint(operators, flavour, path.sector(0), orientation.early_operator(flavour).adjoint, statistics));
    for (int flavour : path.special_operators) {
      endpoints.late.push_back(bind_endpoint(operators, flavour, path.sector(special), orientation.late_operator(flavour).adjoint, statistics));
      if (orientation.reflected())
        endpoints.coefficients.emplace_back(tau_values_reflect(all, all, flavour));
      else
        endpoints.coefficients.emplace_back(tau_values(all, flavour, all));
    }
    return endpoints;
  }

  SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, block_sparse::diagram::BackbonePath const &path,
                                 BlockOpSymQuartet const &operators, nda::array_const_view<nda::dcomplex, 3> tau_values,
                                 nda::array_const_view<nda::dcomplex, 3> tau_values_reflect, std::span<triqs_xca::Statistics const> statistics) {
    int sign = path.backbone.self_energy_sign();
    return {bind::backbone(propagator, operators, path), endpoint_sum(path, operators, tau_values, tau_values_reflect, statistics), path.sector(0),
            sign};
  }
} // namespace triqs_xca::block_sparse::bind
