#pragma once
#include <functional>
#include <type_traits>
#include "diagram/backbone_path.hpp"
#include "operator_support.hpp"

namespace triqs_xca::block_sparse {
  /// Discover sector realizations of a range with fully selected operator vertices.
  /// Unresolved self-energy/external endpoints need their own completion discovery.
  /// Visits start sectors in order and checks masks on newly reached propagator
  /// sectors. The initial sector mask is left to the caller's completion policy,
  /// preserving correlator behavior. Copy callback records to retain them.
  template <triqs_xca::diagram::Kind K>
  void visit_backbone_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Backbone<K> const &backbone,
                            BlockOpSymQuartet const &operators, OperatorSupportTable const &support,
                            triqs_xca::diagram::BackboneRange range,
                            std::type_identity_t<std::function<void(diagram::BackbonePath<K> const &)>> const &visit);
} // namespace triqs_xca::block_sparse
