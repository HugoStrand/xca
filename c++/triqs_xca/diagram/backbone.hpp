#pragma once
#include "backbone_range.hpp"
#include "tau_kernel.hpp"

namespace triqs_xca::diagram {
  /// One vertex in an assigned diagram. A flavour of -1 is an unresolved special
  /// endpoint, to be supplied by the remaining endpoint sum or observable.
  struct Vertex {
    bool barred;
    bool adjoint;
    int flavour;
    int pole_slot;
  };

  /// Ordered operator vertices and local scalar-kernel prescriptions. A missing
  /// edge factor means the identity, including the trailing correlator edge.
  /// Whole-diagram signs and scalar prefactors belong to Diagram, not this chain.
  struct Backbone {
    std::vector<Vertex> vertices;
    TauKernelPlacement kernel_placement;

    BackboneRange range() const { return {0, static_cast<int>(vertices.size()), 0, static_cast<int>(kernel_placement.edges.extent(0))}; }
  };
} // namespace triqs_xca::diagram
