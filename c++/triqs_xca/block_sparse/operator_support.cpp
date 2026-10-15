#include "operator_support.hpp"

#include <algorithm>
#include <stdexcept>

namespace triqs_xca::block_sparse {

OperatorSupport::OperatorSupport(int sectors, std::vector<SectorTransition> components) {
  if (sectors < 0) throw std::invalid_argument("OperatorSupport: negative sector count");
  offsets_.assign(sectors + 1, 0);
  std::sort(components.begin(), components.end(), [](auto a, auto b) {
    return a.source < b.source || (a.source == b.source && a.target < b.target);
  });
  int previous_source = -1, previous_target = -1;
  for (auto [source, target] : components) {
    if (source < 0 || source >= sectors || target < 0 || target >= sectors)
      throw std::invalid_argument("OperatorSupport: sector outside the physical partition");
    if (source == previous_source && target == previous_target) continue;
    ++offsets_[source + 1];
    targets_.push_back(target);
    previous_source = source;
    previous_target = target;
  }
  for (int s = 0; s < sectors; ++s) offsets_[s + 1] += offsets_[s];
}

std::span<int const> OperatorSupport::targets(int source) const {
  if (source < 0 || source + 1 >= offsets_.size()) throw std::out_of_range("OperatorSupport: invalid source sector");
  return std::span<int const>(targets_).subspan(offsets_[source], offsets_[source + 1] - offsets_[source]);
}

int OperatorSupport::single_target(int source) const {
  auto row = targets(source);
  if (row.size() > 1) throw std::invalid_argument("OperatorSupport: current CPU path executor requires a single target per source");
  return row.empty() ? -1 : row.front();
}

namespace {
constexpr auto all = nda::range::all;

template <typename Group, typename Slice>
OperatorSupport describe(Group const &group, Slice slice) {
  std::vector<SectorTransition> components;
  for (int source = 0; source < group.get_num_block_cols(); ++source) {
    int target = group.get_block_index(source);
    if (target < 0) continue;
    auto matrix = slice(group.get_block(source));
    if (std::any_of(matrix.begin(), matrix.end(), [](dcomplex z) { return z != dcomplex{}; }))
      components.push_back({source, target});
  }
  return {group.get_num_block_cols(), std::move(components)};
}
}

OperatorSupportTable::OperatorSupportTable(BlockOpSymQuartet const &operators) : poles(operators.p) {
  for (int flavour = 0; flavour < operators.sym_set_labels.size(); ++flavour) {
    int group = operators.sym_set_labels(flavour), index = operators.sym_set_inds(flavour);
    auto slice = [=](auto block) { return block(index, all, all); };
    plain.push_back(describe(operators.Fs[group], slice));
    adjoint.push_back(describe(operators.F_dags[group], slice));
    for (int pole = 0; pole < poles; ++pole) {
      auto bar_slice = [=](auto block) { return block(index, pole, all, all); };
      barred_adjoint.push_back(describe(operators.F_dag_bars[group], bar_slice));
      reflected_bar.push_back(describe(operators.F_bars_refl[group], bar_slice));
    }
  }
}

OperatorSupport const &OperatorSupportTable::vertex(triqs_xca::diagram::Diagram const &diagram, int vertex) const {
  auto const &v = diagram.vertices.at(vertex);
  if (!v.barred) return (v.adjoint ? adjoint : plain).at(v.flavour);
  int pole = diagram.pole_indices(v.pole_slot);
  return (v.adjoint ? barred_adjoint : reflected_bar).at(v.flavour * poles + pole);
}

} // namespace triqs_xca::block_sparse
