"""Private full-shape evaluator variant; reject changes to production seams."""
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:]);cuda=root/'c++/triqs_xca/block_sparse/cuda'
header=(cuda/'diagram_evaluator.hpp').read_text();source=(cuda/'diagram_evaluator.cpp').read_text()
def once(old,new):
    global source
    if source.count(old)!=1:raise RuntimeError(f'Liveness seam changed: {old[:80]}')
    source=source.replace(old,new)
# Same solves and convolution inputs/shapes. Only storage lifetimes change.
for name,signature in (
 ('edge','    void edge(Tensor &T, std::vector<Tensor> const &G, Backbone &backbone, int e, SelfEnergyPath const &path) {'),
 ('correlator_edge','    void correlator_edge(Tensor &T, TensorConstView source, CorrelatorBackbone &backbone, int e, bool right) {')):
    once(signature,signature+'\n      dlr->release_scratch(); // Drain previous use before creating the next phase.');
old='      Tensor weighted(source.shape()), gc(source.shape()), tc(T.shape()), out(T.shape());'
if source.count(old)!=2:raise RuntimeError('Expected both edge allocation sites')
source=source.replace(old,'      Tensor weighted(source.shape()), gc(source.shape());')
old='      dlr->vals2coefs(weighted, gc);\n      dlr->vals2coefs(T, tc);'
if source.count(old)!=2:raise RuntimeError('Expected both coefficient conversion sites')
source=source.replace(old,'''      dlr->vals2coefs(weighted, gc);
      ops->finish();
      weighted = Tensor(); // No subsequent phase reads the weighted values.
      Tensor tc(T.shape());
      dlr->vals2coefs(T, tc);
      ops->finish();
      T = Tensor(); // All subsequent arithmetic reads the coefficient copy.
      Tensor out(tc.shape());''')
# Edge sampling happens while complete convolution scratch is live. Finish before
# freeing either coefficient input, then release unused scratch before next vertex.
old='      sample_memory();\n      T = std::move(out);'
# This seam is shared with vertices/special paths; patch only two edge function bodies.
for start,end in [('    void edge(', '    double kernel_prefactor('),('    void correlator_edge(', '    Tensor fragment(')]:
    a=source.index(start);b=source.index(end,a);body=source[a:b]
    if body.count(old)!=1:raise RuntimeError('Unexpected edge completion seam')
    body=body.replace(old,'''      sample_memory();
      ops->finish();
      gc = Tensor();
      tc = Tensor();
      dlr->release_scratch();
      T = std::move(out);''')
    source=source[:a]+body+source[b:]
# At most seven temporary rank tensors overlap: three coefficient/output caller
# tensors plus four disjoint convolution scratch tensors. Solves need <= four;
# special fallback <= six; vertices/reflection <= two. RHS and convolution scratch
# cannot overlap. Stored correlator fragments stay in their existing 2*function
# reserve; endpoints and result buffers retain their existing additional reserves.
once('return reference_storage_bound(r, model.dimensions(), dlr_bound, operator_budget, kernel_bytes, mapped_storage_bound());',
     'auto tables = checked_add(checked_multiply(5, tensor_bytes(1, r, r)), checked_multiply(checked_add(r, 1), sizeof(int)));\n'
     '      return reference_storage_bound(r, model.dimensions(), tables, operator_budget, kernel_bytes, mapped_storage_bound());')
once('return generic - (dlr_bound - tables) - inventory_memory(r, model.dimensions(), 0).maximum_rank_buffer;',
     'return generic - inventory_memory(r, model.dimensions(), 0).maximum_rank_buffer;')
# A failed call can leave scratch. Drain it before reserving caller tensors for
# every new admitted call, including correlators. Reject explicit budgets first.
once('      if (needed > budget) throw std::length_error("CUDA evaluator explicit storage exceeds device budget");\n      std::size_t free',
     '      if (needed > budget) throw std::length_error("CUDA evaluator explicit storage exceeds device budget");\n'
     '      dlr->release_scratch();\n      std::size_t free')
header=header.replace('DiagramEvaluator','LifetimeDiagramEvaluator')
source=source.replace('DiagramEvaluator','LifetimeDiagramEvaluator').replace('"diagram_evaluator.hpp"','"lifetime_evaluator.hpp"')
(out/'lifetime_evaluator.hpp').write_text(header);(out/'lifetime_evaluator.cpp').write_text(source)
