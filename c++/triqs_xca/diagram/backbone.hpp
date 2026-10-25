#pragma once
#include "backbone_range.hpp"
#include "tau_kernel.hpp"

namespace triqs_xca::diagram {
  /// A vertex refers to its topology arc, including the distinguished endpoints.
  /// Flavour and pole choices belong to Arcs. Only barred operators need a pole.
  struct Vertex {
    int arc_idx     = -1;
    bool barred     = false;
    bool adjoint    = false;
    int kernel_sign = 0;
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
