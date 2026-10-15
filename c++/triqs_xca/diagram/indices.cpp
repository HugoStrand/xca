#include "indices.hpp"
#include <stdexcept>
#include <string>

namespace triqs_xca::diagram {
  namespace {
    long checked_power(long base, int exponent) {
      long value = 1;
      for (int i = 0; i < exponent; ++i)
        if (__builtin_mul_overflow(value, base, &value)) throw std::overflow_error("assignment count does not fit in a long");
      return value;
    }
    int direction_count(int order, Kind kind) { return order - (kind == Kind::correlator); }
  } // namespace

  long count_diagrams(int order, int flavours, long pole_count, Kind kind) {
    if (order < 1 || flavours < 0 || pole_count < 0) throw std::invalid_argument("invalid assignment space dimensions");
    long count    = checked_power(2, direction_count(order, kind));
    long orbitals = checked_power(flavours, order - 1);
    if (__builtin_mul_overflow(count, orbitals, &count)) throw std::overflow_error("assignment count does not fit in a long");
    for (int p = 0; p < order - 1; ++p)
      if (__builtin_mul_overflow(count, pole_count, &count)) throw std::overflow_error("assignment count does not fit in a long");
    return count;
  }

  Indices decode_indices(int order, int flavours, long pole_count, long flat_index, Kind kind) {
    long count = count_diagrams(order, flavours, pole_count, kind);
    if (flat_index < 0 || flat_index >= count)
      throw std::invalid_argument("flat index " + std::to_string(flat_index) + " is out of range [0, " + std::to_string(count) + ")");
    Indices result{nda::vector<int>(direction_count(order, kind)), nda::vector<int>(order - 1), nda::vector<int>(order - 1)};
    long remaining = flat_index;
    for (int a = 0; a < order - 1; ++a) {
      result.flavours(a) = remaining % flavours;
      remaining /= flavours;
    }
    for (int a = 0; a < order - 1; ++a) {
      result.pole_indices(a) = remaining % pole_count;
      remaining /= pole_count;
    }
    for (int a = 0; a < result.directions.size(); ++a) {
      result.directions(a) = remaining % 2;
      remaining /= 2;
    }
    return result;
  }

} // namespace triqs_xca::diagram
