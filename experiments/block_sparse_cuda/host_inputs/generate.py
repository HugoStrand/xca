"""Host-backed input experiment layered on the qualified lifetime schedule."""
from pathlib import Path
import subprocess
import sys
root,out=map(Path,sys.argv[1:])
subprocess.run([sys.executable,str(root/'experiments/block_sparse_cuda/phase_liveness/generate.py'),str(root),str(out)],check=True)
header=(out/'lifetime_evaluator.hpp').read_text();source=(out/'lifetime_evaluator.cpp').read_text()
def once(old,new):
    global source
    if source.count(old)!=1:raise RuntimeError(f'Host input seam changed: {old[:90]}')
    source=source.replace(old,new)
source=source.replace('std::vector<Tensor> const &G','triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G')
once('auto const &source = G[path.propagator_blocks[e]];', 'auto source = G[path.propagator_blocks[e]].data();')
once('void correlator_edge(Tensor &T, TensorConstView source,', 'void correlator_edge(Tensor &T, nda::array_const_view<Complex,3> source,')
old='      Tensor weighted(source.shape()), gc(source.shape());\n      ops->copy(source, weighted);'
if source.count(old)!=2:raise RuntimeError('Expected both edge input upload sites')
source=source.replace(old,'''      Tensor weighted(source), gc(source.shape());
      ops->wait_for_default_uploads(); // Owned-stream handoff after host upload.''')
once('auto const &source = G[left ? path.end_block : path.root_block];\n      Tensor T(source.shape());\n      ops->copy(source, T);',
     'auto source = G[left ? path.end_block : path.root_block].data();\n      Tensor T(source);\n      ops->wait_for_default_uploads();')
source=source.replace('correlator_edge(T, G[path.propagator_blocks[v]],', 'correlator_edge(T, G[path.propagator_blocks[v]].data(),')
source=source.replace('correlator_edge(T, G[source_block],', 'correlator_edge(T, G[source_block].data(),')
once('      std::vector<Tensor> input(G.size());\n      // Unlike self-energy, a masked starting root can contribute. Keep every\n      // stored input value, including tiny blocks below the whole-block mask.\n      for (long b = 0; b < G.size(); ++b) input[b] = Tensor(G[b].data());',
     '      auto input=G; // Borrow exact host values for the synchronous call, including tiny roots.')
once('auto const &first = G[path.propagator_blocks[0]];\n      Tensor T(first.shape());\n      ops->copy(first, T);',
     'auto first = G[path.propagator_blocks[0]].data();\n      Tensor T(first);\n      ops->wait_for_default_uploads();')
once('    std::vector<Tensor> input(G.size()), output(G.size());','    auto input=G;\n    std::vector<Tensor> output(G.size());')
once('        if (live[b]) input[b] = Tensor(G[b].data());','        // Inputs are borrowed from host and uploaded only at their next use.')
once('return reference_storage_bound(r, model.dimensions(), tables, operator_budget, kernel_bytes, mapped_storage_bound());',
     'return reference_storage_bound(r, model.dimensions(), tables, operator_budget, kernel_bytes, mapped_storage_bound()) - function_bytes(r, model.dimensions());')
# The separate optimized arena still uploads whole input/output. Do not claim its
# storage or silently ignore its explicit policy in this host-input experiment.
once('      if (nca_selected != NcaBackend::Reference && nca_selected != NcaBackend::RealBatch)',
     '      if (nca_selected != NcaBackend::Reference) throw std::invalid_argument("Host input experiment requires reference NCA");\n      if (nca_selected != NcaBackend::Reference && nca_selected != NcaBackend::RealBatch)')
header=header.replace('LifetimeDiagramEvaluator','HostInputDiagramEvaluator')
source=source.replace('LifetimeDiagramEvaluator','HostInputDiagramEvaluator').replace('"lifetime_evaluator.hpp"','"host_input_evaluator.hpp"')
(out/'host_input_evaluator.hpp').write_text(header);(out/'host_input_evaluator.cpp').write_text(source)
