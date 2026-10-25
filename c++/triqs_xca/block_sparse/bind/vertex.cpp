#include "vertex.hpp"

namespace triqs_xca::block_sparse::bind {
  namespace {
    constexpr auto all = nda::range::all;
  }

  nda::array_const_view<dcomplex, 2> vertex(BlockOpSymQuartet const &operators, triqs_xca::diagram::Diagram const &diagram, int vertex,
                                            int source_sector) {
    auto const &v = diagram.backbone.vertices[vertex];
    int set = operators.sym_set_labels(diagram.arcs[v.arc_idx].retained_flavour), index = operators.sym_set_inds(diagram.arcs[v.arc_idx].retained_flavour);
    if (v.barred) {
      int pole       = diagram.arcs[v.arc_idx].pole_index;
      auto const &op = v.adjoint ? operators.F_dag_bars[set] : operators.F_bars_refl[set];
      return op.get_block(source_sector)(index, pole, all, all);
    }
    auto const &op = v.adjoint ? operators.F_dags[set] : operators.Fs[set];
    return op.get_block(source_sector)(index, all, all);
  }

} // namespace triqs_xca::block_sparse::bind
