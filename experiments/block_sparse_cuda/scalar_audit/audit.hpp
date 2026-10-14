#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
namespace scalar_cutoff_audit {
struct Stats {
  std::size_t lanes = 0, mismatches = 0, outside_band = 0, nonfinite = 0;
  double magnitude_error = 0, value_error = 0;
};
inline thread_local Stats stats;
inline thread_local bool repair_all = false;
inline void observe(double batched, double reference, double error, bool in_band, bool finite) {
  ++stats.lanes;
  if (!finite || !std::isfinite(batched) || !std::isfinite(reference) || !std::isfinite(error)) ++stats.nonfinite;
  stats.magnitude_error = std::max(stats.magnitude_error, std::abs(batched - reference));
  stats.value_error = std::max(stats.value_error, error);
  if ((batched > 1e-16) != (reference > 1e-16)) {
    ++stats.mismatches;
    if (!in_band) ++stats.outside_band;
  }
}
}
