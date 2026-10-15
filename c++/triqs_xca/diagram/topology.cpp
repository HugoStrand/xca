#include <algorithm>
#include "triqs_xca/diagram/topology.hpp"

#include <nda/layout_transforms.hpp>

#include <string>
#include <vector>

namespace triqs_xca::diagram::topology {

  int permutation_parity(nda::array_const_view<int, 1> perm) {

    int n = perm.extent(0);

    // Check that perm is a genuine permutation of {0, ..., n-1}.
    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; ++i) {
      int p = perm(i);
      if (p < 0 || p >= n) {
        throw std::invalid_argument("permutation_parity: entry " + std::to_string(p) + " is out of range [0, " + std::to_string(n) + ")");
      }
      if (seen[p]) { throw std::invalid_argument("permutation_parity: entry " + std::to_string(p) + " appears more than once"); }
      seen[p] = true;
    }

    // Walk the cycles. A cycle of length L is an even permutation when L is odd
    // (it decomposes into L-1 transpositions), so accumulate (L-1) mod 2 over
    // all cycles, i.e. (n - number_of_cycles) mod 2.
    std::vector<bool> visited(n, false);
    int n_transpositions = 0;
    for (int i = 0; i < n; ++i) {
      if (visited[i]) continue;
      int j           = i;
      int cycle_len   = 0;
      while (!visited[j]) {
        visited[j] = true;
        j          = perm(j);
        ++cycle_len;
      }
      n_transpositions += cycle_len - 1;
    }

    return (n_transpositions % 2 == 0) ? 1 : -1;
  }

  int topology_parity(nda::array_const_view<int, 2> topology) {
    nda::array<int, 1> perm = nda::flatten(topology);
    return permutation_parity(perm);
  }

  nda::array<int, 2> fermionic_topology(nda::array_const_view<int, 2> topology, nda::array_const_view<bool, 1> fermionic) {

    int n_rows = topology.extent(0);
    int n_cols = topology.extent(1);
    long n_elem = topology.size();

    if (fermionic.extent(0) != n_elem) {
      throw std::invalid_argument("fermionic_topology: fermionic flag vector length (" + std::to_string(fermionic.extent(0))
                                  + ") must equal the number of topology elements (" + std::to_string(n_elem) + ")");
    }

    // Collect the rows whose vertices are all fermionic, and record which
    // vertices survive in the output. Note that dropping a pair removes both of
    // its vertices, even a fermionic one paired with a bosonic partner.
    std::vector<int> kept_rows;
    std::vector<bool> present(n_elem, false);
    for (int r = 0; r < n_rows; ++r) {
      bool all_fermionic = true;
      for (int c = 0; c < n_cols; ++c) {
        int v = topology(r, c);
        if (v < 0 || v >= n_elem) {
          throw std::invalid_argument("fermionic_topology: topology entry " + std::to_string(v) + " is out of range [0, " + std::to_string(n_elem)
                                      + ")");
        }
        if (!fermionic(v)) {
          all_fermionic = false;
          break;
        }
      }
      if (all_fermionic) {
        kept_rows.push_back(r);
        for (int c = 0; c < n_cols; ++c) { present[topology(r, c)] = true; }
      }
    }

    // Compact the surviving vertex labels to a contiguous range 0, ..., m-1,
    // where m is the number of surviving vertices. Each surviving vertex maps to
    // its rank among the survivors, which equals its original label minus the
    // number of removed vertices below it. This keeps the output a valid
    // permutation of {0, ..., m-1}.
    std::vector<int> new_label(n_elem, -1);
    int next = 0;
    for (long v = 0; v < n_elem; ++v) {
      if (present[v]) new_label[v] = next++;
    }

    // Build the fermionic-only topology with the compacted vertex labels.
    auto result = nda::array<int, 2>(static_cast<long>(kept_rows.size()), n_cols);
    for (long i = 0; i < static_cast<long>(kept_rows.size()); ++i) {
      for (int c = 0; c < n_cols; ++c) { result(i, c) = new_label[topology(kept_rows[i], c)]; }
    }
    return result;
  }

} // namespace triqs_xca::diagram::topology

namespace triqs_xca::diagram {

Topology::Topology(nda::array<int, 2> input) : arcs(std::move(input)) {
  int m = order();
  if (m < 1 || arcs.extent(1) != 2) throw std::invalid_argument("topology must contain at least one pair of vertices");
  std::vector<int> vertices(2 * m);
  for (int a = 0; a < m; ++a) {
    if (a > 0 && arcs(a, 0) <= arcs(a - 1, 0)) throw std::invalid_argument("first pair indices must be in increasing order");
    if (arcs(a, 0) >= arcs(a, 1)) throw std::invalid_argument("first row of topology must contain smaller-numbered vertices");
    vertices[2 * a] = arcs(a, 0);
    vertices[2 * a + 1] = arcs(a, 1);
  }
  std::ranges::sort(vertices);
  for (int v = 0; v < 2 * m; ++v)
    if (vertices[v] != v) throw std::invalid_argument("topology must contain each vertex index from 0 to 2m-1 exactly once");
}

} // namespace triqs_xca::diagram
