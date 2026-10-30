#include "arcs.hpp"
#include <cmath>
#include <limits>
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

    void validate_choices(Arcs const &arcs, int order, Kind kind) {
      if (order < 1 || arcs.size() != static_cast<std::size_t>(order)) throw std::invalid_argument("wrong number of arcs");
      if (arcs[0].retained_flavour != -1 || arcs[0].pole_index != -1)
        throw std::invalid_argument("special arc has no retained flavour or selected pole");
      if (kind == Kind::correlator && arcs[0].direction != Direction::external) throw std::invalid_argument("correlator endpoints must be external");
      int first = kind == Kind::correlator ? 1 : 0;
      for (int a = first; a < order; ++a)
        if (arcs[a].direction != Direction::backward && arcs[a].direction != Direction::forward)
          throw std::invalid_argument("internal and self-energy arc directions must be forward or backward");
      for (int a = 1; a < order; ++a)
        if (arcs[a].retained_flavour < 0 || arcs[a].pole_index < 0) throw std::invalid_argument("negative internal arc index");
    }

    Arcs decode_choices(int order, int flavours, long pole_count, long flat_index, Kind kind) {
      long count = count_diagrams(order, flavours, pole_count, kind);
      if (flat_index < 0 || flat_index >= count)
        throw std::invalid_argument("flat index " + std::to_string(flat_index) + " is out of range [0, " + std::to_string(count) + ")");
      Arcs result(order);
      long remaining = flat_index;
      for (int a = 0; a < order - 1; ++a) {
        result[a + 1].retained_flavour = remaining % flavours;
        remaining /= flavours;
      }
      for (int a = 0; a < order - 1; ++a) {
        result[a + 1].pole_index = remaining % pole_count;
        remaining /= pole_count;
      }
      int first = kind == Kind::correlator ? 1 : 0;
      for (int a = first; a < order; ++a) {
        result[a].direction = static_cast<Direction>(remaining % 2);
        remaining /= 2;
      }
      return result;
    }

  } // namespace

  void validate_arcs(Arcs const &arcs, int order) {
    if (arcs.empty()) throw std::invalid_argument("wrong number of arcs");
    validate_choices(arcs, order, arcs[0].direction == Direction::external ? Kind::correlator : Kind::self_energy);
    for (std::size_t a = 1; a < arcs.size(); ++a) {
      if (arcs[a].statistics != Statistics::fermionic && arcs[a].statistics != Statistics::bosonic)
        throw std::invalid_argument("invalid arc operator statistics");
      if (arcs[a].pole_sign < -1 || arcs[a].pole_sign > 1) throw std::invalid_argument("invalid arc pole sign");
    }
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

  Arcs prepare_arcs(Arcs choices, std::span<Statistics const> statistics, nda::vector_const_view<double> poles, Kind kind) {
    if (kind != Kind::self_energy && kind != Kind::correlator) throw std::invalid_argument("invalid arc preparation kind");
    if (choices.size() > std::numeric_limits<int>::max()) throw std::invalid_argument("too many arcs");
    validate_choices(choices, static_cast<int>(choices.size()), kind);
    choices[0].statistics = Statistics::bosonic; // unresolved endpoint parity; not used by preparation
    choices[0].pole_sign  = 0;
    for (std::size_t a = 1; a < choices.size(); ++a) {
      auto &arc = choices[a];
      if (arc.retained_flavour >= statistics.size()) throw std::invalid_argument("arc flavour index out of range");
      if (arc.pole_index >= poles.size()) throw std::invalid_argument("arc pole index out of range");
      arc.statistics = statistics[arc.retained_flavour];
      if (arc.statistics != Statistics::bosonic && arc.statistics != Statistics::fermionic)
        throw std::invalid_argument("invalid arc operator statistics");
      double pole = poles(arc.pole_index);
      if (!std::isfinite(pole)) throw std::invalid_argument("selected arc pole must be finite");
      arc.pole_sign = (pole > 0) - (pole < 0);
    }
    return choices;
  }

  Arcs decode_arcs(int order, std::span<Statistics const> statistics, nda::vector_const_view<double> poles, long flat_index, Kind kind) {
    if (statistics.size() > std::numeric_limits<int>::max()) throw std::invalid_argument("too many operator flavours");
    return prepare_arcs(decode_choices(order, static_cast<int>(statistics.size()), poles.size(), flat_index, kind), statistics, poles, kind);
  }

} // namespace triqs_xca::diagram
