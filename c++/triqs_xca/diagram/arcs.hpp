#pragma once
#include "triqs_xca/action.hpp"
#include <span>
#include <vector>

namespace triqs_xca::diagram {

  enum class Direction { external = -1, backward = 0, forward = 1 };

  /// One arc choice. Arc zero has no retained flavour or pole; for a correlator
  /// it denotes the external operator pair rather than an interaction.
  struct Arc {
    Direction direction                = Direction::external;
    int retained_flavour               = -1;
    int pole_index                     = -1;
    Statistics statistics              = Statistics::bosonic;
    int pole_sign                      = 0;
    bool operator==(Arc const &) const = default;
  };

  enum class Kind { self_energy, correlator };

  /// Arc choices indexed by connectivity-array row, including the special row zero.
  using Arcs = std::vector<Arc>;

  /// Validate enriched choices without consulting their original Action.
  void validate_arcs(Arcs const &arcs, int order);
  long count_diagrams(int order, int flavours, long pole_count, Kind kind);

  /// Resolve selected statistics and three-way pole signs. Special-arc metadata
  /// is unresolved and never indexed. Existing metadata in choices is replaced.
  /// A barred contraction must couple operators of the retained flavour's parity.
  Arcs prepare_arcs(Arcs choices, std::span<Statistics const> statistics, nda::vector_const_view<double> poles, Kind kind);
  Arcs decode_arcs(int order, std::span<Statistics const> statistics, nda::vector_const_view<double> poles, long flat_index, Kind kind);

  template <typename Storage> Arcs prepare_arcs(Arcs choices, Action<Storage> const &action, Kind kind) {
    return prepare_arcs(std::move(choices), action.statistics, action.poles, kind);
  }

  template <typename Storage> Arcs decode_arcs(int order, Action<Storage> const &action, long flat_index, Kind kind) {
    return decode_arcs(order, action.statistics, action.poles, flat_index, kind);
  }
} // namespace triqs_xca::diagram
