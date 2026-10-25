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
    VertexTauKernel kernel;
  };

  struct Edge {
    std::vector<KernelFactor> kernels; // increasing arc order; an empty list is identity
  };

  /// Ordered operator vertices and local scalar-kernel prescriptions. A missing
  /// edge factor means the identity, including the trailing correlator edge.
  /// The normalization recipe belongs to the whole backbone, not each segment.
  struct Backbone {
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    TauKernelPrefactor kernel_prefactor;

    BackboneRange range() const { return {0, static_cast<int>(vertices.size()), 0, static_cast<int>(edges.size())}; }
  };
  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind);
} // namespace triqs_xca::diagram
