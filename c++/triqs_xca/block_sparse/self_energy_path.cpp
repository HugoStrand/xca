#include "self_energy_path.hpp"

#include <stdexcept>

namespace triqs_xca::block_sparse {

void visit_self_energy_paths(BlockDiagOpFun const &Gt, DiagramAssignment const &assignment, BlockOpSymQuartet const &Fq, OperatorSupports const &support,
                             SelfEnergyPath &path, std::function<void(SelfEnergyPath const &)> const &visit) {

  int m = assignment.topology.order();
  int vct0 = assignment.topology.special_vertex();

  if (path.propagator_blocks.size() != 2 * m - 1 || path.dimensions.size() != 2 * m + 1)
    throw std::invalid_argument("visit_self_energy_paths: workspace does not match diagram order");
  path.flat_index = assignment.flat_index;
  path.special_vertex = vct0;
  auto &ind_path = path.propagator_blocks;
  auto &block_dims = path.dimensions;
  int q = static_cast<int>(Fq.Fs.size());

  /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix
          3            3            2            2            1            1            0
      --------     --------     --------     --------     --------     --------     --------
    4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
      |      |     |      |     |      |     |      |     |      |     |      |     |      |
      --------     --------     --------     --------     --------     --------     --------
  */

  auto F_selector_p_ix = [&](int w, int p_ix) -> BlockOpSymSet const & { return assignment.vertices[w].dagger ? Fq.F_dags[p_ix] : Fq.Fs[p_ix]; };
  auto F_selector      = [&](int w) -> BlockOpSymSet const & { return F_selector_p_ix(w, Fq.sym_set_labels(assignment.vertices[w].orbital)); };

  bool incomplete_path    = false;
  auto is_path_incomplete = [&](int w, int ip) { return (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)); };

  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of self-energy

    for (int p_kap = 0; p_kap < q; p_kap++) { // loop over symmetry sets on the zero vertex

      // traverse factors in two halves
      // -----------------------------------------------------------------------------------
      // first half: all vertices before vertex connected to zero

      // ind_path setup 1

      int ip = -1; // running block index

      for (int w = 0; w < vct0; w++) {
        ip              = (w == 0) ? F_selector_p_ix(w, p_kap).get_block_index(b_ix) : support.vertex(assignment, w).single_target(ip);
        incomplete_path = is_path_incomplete(w, ip);
        if (incomplete_path) break;
        ind_path(w) = ip;
      }

      if (incomplete_path) continue;

      // block_dims setup 1

      block_dims(0) = F_selector_p_ix(0, p_kap).get_block_size(b_ix, 1);
      block_dims(1) = F_selector_p_ix(0, p_kap).get_block_size(b_ix, 0);

      for (int w = 0; w < vct0 - 1; w++) { block_dims(w + 2) = F_selector(w + 1).get_block_size(ind_path(w), 0); }

      // -----------------------------------------------------------------------------------
      // second half: vertex connected to zero and above

      for (int p_mu = 0; p_mu < q; p_mu++) { // loop over symmetry sets on the vertex connected to vertex 0

        // Coefficient validation requires exact zeros between different groups.
        // Only endpoints from the same execution group can contribute.
        if (p_mu != p_kap) continue;

        // ind_path setup 2

        ip = ind_path(vct0 - 1);

        for (int w = vct0; w < 2 * m; w++) {
          ip              = (w == vct0) ? F_selector_p_ix(w, p_mu).get_block_index(ip) : support.vertex(assignment, w).single_target(ip);
          incomplete_path = is_path_incomplete(w, ip);
          if (incomplete_path) break;
          if (w < 2 * m - 1) ind_path(w) = ip;
        }

        if (incomplete_path) continue;

        // block_dims setup 2

        block_dims(vct0 + 1) = F_selector_p_ix(vct0, p_mu).get_block_size(ind_path(vct0 - 1), 0);

        for (int w = vct0; w < 2 * m - 1; w++) { block_dims(w + 2) = F_selector(w + 1).get_block_size(ind_path(w), 0); }

        path.output_sector = b_ix;
        path.terminal_sector = ip;
        path.kap_set = p_kap;
        path.mu_set = p_mu;
        visit(path);
      }
    }
  }
}

} // namespace triqs_xca::block_sparse
