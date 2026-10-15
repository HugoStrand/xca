#include "diagram_model.hpp"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace triqs_xca {

InteractionModel::InteractionModel(int flavours, std::vector<InteractionFamily> families) : families_(std::move(families)) {
  if (flavours < 0) throw std::invalid_argument("negative number of interaction flavours");
  statistics_.resize(flavours);
  std::vector<bool> assigned(flavours, false);
  for (auto const &family : families_) {
    for (int flavour : family.flavours) {
      if (flavour < 0 || flavour >= flavours) throw std::invalid_argument("interaction family flavour is out of range");
      if (assigned[flavour]) throw std::invalid_argument("interaction flavour belongs to multiple families");
      assigned[flavour] = true;
      statistics_[flavour] = family.statistics;
      if (family.kind == InteractionKind::hybridization) hybridization_flavours_.push_back(flavour);
    }
  }
  for (bool present : assigned)
    if (!present) throw std::invalid_argument("interaction flavour has no family");
}

InteractionModel hybridization_retarded_model(int hybridization_flavours, int retarded_flavours) {
  if (hybridization_flavours < 0 || retarded_flavours < 0)
    throw std::invalid_argument("negative number of interaction flavours");
  if (hybridization_flavours > std::numeric_limits<int>::max() - retarded_flavours)
    throw std::overflow_error("too many interaction flavours");
  std::vector<int> hybridization(hybridization_flavours), retarded(retarded_flavours);
  std::iota(hybridization.begin(), hybridization.end(), 0);
  std::iota(retarded.begin(), retarded.end(), hybridization_flavours);
  return {hybridization_flavours + retarded_flavours,
          {{InteractionKind::hybridization, Statistics::fermionic, std::move(hybridization)},
           {InteractionKind::retarded, Statistics::bosonic, std::move(retarded)}}};
}

ObservableBoundary uniform_observable(bool fermionic) {
  auto statistics = fermionic ? Statistics::fermionic : Statistics::bosonic;
  return {statistics, statistics};
}

} // namespace triqs_xca
