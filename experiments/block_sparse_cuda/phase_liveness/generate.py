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
# Existing public bounds intentionally remain conservative until measured and audited.
header=header.replace('DiagramEvaluator','LifetimeDiagramEvaluator')
source=source.replace('DiagramEvaluator','LifetimeDiagramEvaluator').replace('"diagram_evaluator.hpp"','"lifetime_evaluator.hpp"')
(out/'lifetime_evaluator.hpp').write_text(header);(out/'lifetime_evaluator.cpp').write_text(source)
