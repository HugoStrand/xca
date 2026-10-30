#include "backbone_paths.hpp"
#include <stdexcept>

namespace triqs_xca::block_sparse {
  void visit_backbone_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Backbone const &backbone,
                            BlockOpSymQuartet const &operators, OperatorSupportTable const &support,
                            triqs_xca::diagram::BackboneRange range,
                            std::function<void(block_sparse::diagram::BackbonePath const &)> const &visit) {
    range.validate();
    if (!backbone.range().contains(range)) throw std::invalid_argument("backbone range is outside backbone");
    if (range.empty()) return; // no propagator or operator contribution to visit
    for (int v = range.first_vertex; v < range.vertex_end; ++v)
      if (v == 0 || v == backbone.endpoints.special_vertex) throw std::invalid_argument("backbone range contains an unresolved endpoint");
    block_sparse::diagram::BackbonePath path(range);
    auto op = [&](int vertex) -> BlockOpSymSet const & {
      auto const &v = backbone.vertices[vertex];
      int set = operators.sym_set_labels(v.flavour);
      return v.adjoint ? operators.F_dags[set] : operators.Fs[set];
    };
    for (int start = 0; start < propagator.get_num_block_cols(); ++start) {
      int sector = start;
      bool incomplete = false;
      path.sector(range.first_vertex) = sector;
      path.dimension(range.first_vertex) = propagator.get_block_size(start);
      for (int v = range.first_vertex; v < range.vertex_end; ++v) {
        int source = sector;
        sector = support.vertex(backbone.vertices[v]).single_target(source);
        bool has_edge = range.first_edge <= v && v < range.edge_end;
        if (sector == -1 || (has_edge && propagator.get_zero_block_index(sector) == -1)) {
          incomplete = true;
          break;
        }
        path.sector(v + 1) = sector;
        path.dimension(v + 1) = op(v).get_block_size(source, 0);
      }
      if (!incomplete) visit(path);
    }
  }
} // namespace triqs_xca::block_sparse
