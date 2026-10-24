#include "action.hpp"

namespace triqs_xca::block_sparse {
  nda::vector<long> operator_block_dims(BlockOpSymQuartet const &operators) {
    if (operators.Fs.empty()) throw std::invalid_argument("Action: no operator symmetry sets");
    nda::vector<long> dims(operators.Fs.front().get_num_block_cols());
    dims = -1;
    for (auto const *sets : {&operators.Fs, &operators.F_dags})
      for (auto const &op : *sets)
        for (long b = 0; b < dims.size(); ++b)
          if (op.get_block_index(b) >= 0) dims(b) = op.get_block_size(b, 1);
    return dims;
  }

  void check_operator_block_dims(BlockOpSymQuartet const &operators, nda::vector_const_view<long> dims) {
    for (auto const *sets : {&operators.Fs, &operators.F_dags})
      for (auto const &op : *sets) {
        if (op.get_num_block_cols() != dims.size()) throw std::invalid_argument("Action: operator sector count differs from dimensions");
        for (long b = 0; b < dims.size(); ++b) {
          int target = op.get_block_index(b);
          if (target < 0) continue;
          if (target >= dims.size() || (dims(b) >= 0 && dims(b) != op.get_block_size(b, 1))
              || (dims(target) >= 0 && dims(target) != op.get_block_size(b, 0)))
            throw std::invalid_argument("Action: operator block shape differs from sector dimensions");
        }
      }
  }
}
