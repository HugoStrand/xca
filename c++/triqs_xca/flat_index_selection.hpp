#pragma once

#include <optional>
#include <stdexcept>
#include <nda/nda.hpp>

namespace triqs_xca {

/// A bounded selection at the legacy linear-index boundary. Range storage is
/// constant in the number of assignments; an explicit vector is borrowed only
/// for the duration of the call. Each index is decoded into logical work by the
/// consumer. This type does not prescribe block paths or a numerical schedule.
class FlatIndexSelection {
  long first_ = 0, last_ = 0;
  std::optional<nda::array_const_view<long, 1>> indices_;

public:
  explicit FlatIndexSelection(nda::array_const_view<long, 1> indices) : indices_(indices) {}
  FlatIndexSelection(long first, long last, long count) : first_(first), last_(last) {
    if (first < 0 || last < first || last > count)
      throw std::invalid_argument("assignment range must satisfy 0 <= first <= last <= count");
  }

  bool is_range() const { return !indices_; }

  template <typename Visitor> void for_each(Visitor &&visit) const {
    if (indices_) {
      for (long index : *indices_) visit(index);
    } else {
      for (long index = first_; index < last_; ++index) visit(index);
    }
  }
};

} // namespace triqs_xca
