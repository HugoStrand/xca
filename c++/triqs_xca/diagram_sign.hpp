#pragma once

#include <span>
#include <vector>

#include "diagram_assignment.hpp"

namespace triqs_xca {

enum class Statistics { bosonic, fermionic };

/// Model-boundary adapter for the current hybridization-then-interaction layout.
/// Preparation itself accepts any ordering of explicit statistics.
std::vector<Statistics> fermion_boson_statistics(int fermions, int bosons);

/// Permutation parity after removing every arc with a bosonic endpoint.
/// Order, direction, kernel, and Python observable prefactors are separate.
int permutation_sign(DiagramTopology const &topology, std::span<Statistics const> vertices);

/// The unresolved special arc has only two parity cases, independent of the
/// number of endpoint orbitals. No orbital-pair enumeration is needed here.
struct EndpointSigns {
  int with_fermionic_arc;
  int without_fermionic_arc;

  int operator()(Statistics at_zero, Statistics at_special) const {
    return at_zero == Statistics::fermionic && at_special == Statistics::fermionic ? with_fermionic_arc : without_fermionic_arc;
  }
};

EndpointSigns prepare_endpoint_signs(DiagramAssignment const &assignment, std::span<Statistics const> interactions);

} // namespace triqs_xca
