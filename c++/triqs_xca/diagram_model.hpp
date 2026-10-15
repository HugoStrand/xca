#pragma once

#include <span>
#include <vector>

namespace triqs_xca {

enum class Statistics { bosonic, fermionic };
enum class InteractionKind { hybridization, retarded };

/// A family groups logical interaction flavours, independently of block
/// symmetries. A flavour indexes its operator and the scalar coefficient axes.
/// The list need not be contiguous; its order defines external flavour order
/// for hybridization families. Matrix storage and pole tables live elsewhere.
struct InteractionFamily {
  InteractionKind kind;
  Statistics statistics;
  std::vector<int> flavours;
};

/// Owned model metadata. Every flavour belongs to exactly one family; arithmetic
/// reads explicit statistics and external-flavour lists, not n_hyb/n_int ranges.
class InteractionModel {
  std::vector<InteractionFamily> families_;
  std::vector<Statistics> statistics_;
  std::vector<int> hybridization_flavours_;

public:
  InteractionModel(int flavours, std::vector<InteractionFamily> families);
  std::span<InteractionFamily const> families() const { return families_; }
  std::span<Statistics const> statistics() const { return statistics_; }
  std::span<int const> hybridization_flavours() const { return hybridization_flavours_; }
};

/// Adapter for the existing atom-diag storage: hybridization operators first,
/// followed by retarded-interaction operators. No executor assumes this order.
InteractionModel hybridization_retarded_model(int hybridization_flavours, int retarded_flavours);

/// External operator lists have homogeneous statistics at each endpoint. These
/// are observable metadata, not an interaction flavour or sentinel orbital.
struct ObservableBoundary {
  Statistics at_zero = Statistics::fermionic;
  Statistics at_tau = Statistics::fermionic;
};

/// Compatibility adapter for the existing uniform-statistics correlator API.
ObservableBoundary uniform_observable(bool fermionic);

} // namespace triqs_xca
