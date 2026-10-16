#include "arcs.hpp"
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

  void Arcs::validate(int order, Kind kind) const {
    if (order < 1 || directions.size() != order || pole_indices.size() != order - 1 || flavours.size() != order - 1)
      throw std::invalid_argument("wrong arc assignment dimensions");
    int first = kind == Kind::correlator ? 1 : 0;
    if (first == 1 && directions(0) != -1) throw std::invalid_argument("external correlator arc must be absent");
    for (int a = first; a < order; ++a)
      if (directions(a) != 0 && directions(a) != 1) throw std::invalid_argument("arc direction must be zero or one");
    for (int a = 0; a < order - 1; ++a)
      if (pole_indices(a) < 0 || flavours(a) < 0) throw std::invalid_argument("negative internal arc index");
  }

  long count_diagrams(int order, int flavours, long pole_count, Kind kind) {
    if (order < 1 || flavours < 0 || pole_count < 0) throw std::invalid_argument("invalid assignment space dimensions");
    long count    = checked_power(2, direction_count(order, kind));
    long orbitals = checked_power(flavours, order - 1);
    if (__builtin_mul_overflow(count, orbitals, &count)) throw std::overflow_error("assignment count does not fit in a long");
    for (int p = 0; p < order - 1; ++p)
      if (__builtin_mul_overflow(count, pole_count, &count)) throw std::overflow_error("assignment count does not fit in a long");
    return count;
  }

  Arcs decode_arcs(int order, int flavours, long pole_count, long flat_index, Kind kind) {
    long count = count_diagrams(order, flavours, pole_count, kind);
    if (flat_index < 0 || flat_index >= count)
      throw std::invalid_argument("flat index " + std::to_string(flat_index) + " is out of range [0, " + std::to_string(count) + ")");
    Arcs result{nda::vector<int>(order), nda::vector<int>(order - 1), nda::vector<int>(order - 1)};
    long remaining = flat_index;
    for (int a = 0; a < order - 1; ++a) {
      result.flavours(a) = remaining % flavours;
      remaining /= flavours;
    }
    for (int a = 0; a < order - 1; ++a) {
      result.pole_indices(a) = remaining % pole_count;
      remaining /= pole_count;
    }
    int first            = kind == Kind::correlator ? 1 : 0;
    result.directions(0) = -1;
    for (int a = first; a < order; ++a) {
      result.directions(a) = remaining % 2;
      remaining /= 2;
    }
    return result;
  }

} // namespace triqs_xca::diagram
