#pragma once

#include <span>
#include <vector>

namespace triqs_xca::diagram {

enum class Statistics { bosonic, fermionic };
enum class ArcKind { hybridization, retarded };

/// A family groups logical interaction flavours, independently of block
/// symmetries. A flavour indexes its operator and the scalar coefficient axes.
/// The list need not be contiguous; its order defines external flavour order
/// for hybridization families. Matrix storage and pole tables live elsewhere.
struct ArcFamily {
  ArcKind kind;
  Statistics statistics;
  std::vector<int> flavours;
};

/// Owned model metadata. Every flavour belongs to exactly one family; arithmetic
/// reads explicit statistics and external-flavour lists, not n_hyb/n_int ranges.
class ArcCatalog {
  std::vector<ArcFamily> families_;
  std::vector<Statistics> statistics_;
  std::vector<int> hybridization_flavours_;

public:
  ArcCatalog(int flavours, std::vector<ArcFamily> families);
  std::span<ArcFamily const> families() const { return families_; }
  std::span<Statistics const> statistics() const { return statistics_; }
  std::span<int const> hybridization_flavours() const { return hybridization_flavours_; }
};

/// Adapter for the existing atom-diag storage: hybridization operators first,
/// followed by retarded-interaction operators. No executor assumes this order.
ArcCatalog hybridization_retarded_catalog(int hybridization_flavours, int retarded_flavours);

} // namespace triqs_xca::diagram
