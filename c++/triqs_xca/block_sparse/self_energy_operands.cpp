#include "self_energy_operands.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace triqs_xca::block_sparse {
namespace { constexpr auto all = nda::range::all; }
bool select_diagonal_self_energy(SelfEnergyOperands const &contribution) {
  auto const &path = contribution.backbone.path;
  if (path.sector(path.vertex_end) == contribution.output_sector) return true;
  // Inspect already-bound input coefficients, never the evaluated result. A
  // candidate group path can be inactive even though its internal vertices exist.
  auto const &endpoints = contribution.endpoints;
  for (int late = 0; late < endpoints.late.size(); ++late) {
    for (auto const &early : endpoints.early) {
      auto values = endpoints.coefficients[late](all, early.coefficient_index);
      if (std::any_of(values.begin(), values.end(), [](nda::dcomplex z) { return z != nda::dcomplex{}; }))
        throw std::invalid_argument("block_sparse self-energy: path from sector " + std::to_string(contribution.output_sector)
                                    + " ends in sector " + std::to_string(path.sector(path.vertex_end))
                                    + "; this partition does not close the indexed self-energy contribution. "
                                      "Use a coarser Hilbert-space partition for block-diagonal output.");
    }
  }
  return false;
}


} // namespace triqs_xca::block_sparse
