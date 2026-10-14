#include "../block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  return block_sparse_bench::run<triqs_xca::block_sparse::DiagramEvaluator>(
     argc, argv, "cpu", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<triqs_xca::block_sparse::DiagramEvaluator>(poles, coeff, mesh, ad, ops, dyn);
     });
}
