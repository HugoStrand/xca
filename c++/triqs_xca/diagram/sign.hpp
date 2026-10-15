#pragma once

#include <span>
#include <vector>

#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/diagram/arc_catalog.hpp"

namespace triqs_xca::diagram {

/// Permutation parity after removing every arc with a bosonic endpoint.
/// Order, direction, kernel, and Python observable prefactors are separate.
int permutation_sign(Topology const &topology, std::span<Statistics const> vertices);

/// The unresolved special arc has only two parity cases, independent of the
/// number of endpoint orbitals. No orbital-pair enumeration is needed here.
struct PermutationSigns {
  int with_fermionic_arc;
  int without_fermionic_arc;

  int operator()(Statistics at_zero, Statistics at_special) const {
    return at_zero == Statistics::fermionic && at_special == Statistics::fermionic ? with_fermionic_arc : without_fermionic_arc;
  }
};

PermutationSigns prepare_permutation_signs(Diagram const &assignment, std::span<Statistics const> interactions);

} // namespace triqs_xca::diagram
