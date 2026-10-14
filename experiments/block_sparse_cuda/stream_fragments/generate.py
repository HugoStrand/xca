"""Bound correlator residency with two per-index fragment cache slots."""
from pathlib import Path
import subprocess
import sys
root,out=map(Path,sys.argv[1:])
subprocess.run([sys.executable,str(root/'experiments/block_sparse_cuda/host_inputs/generate.py'),str(root),str(out)],check=True)
header=(out/'host_input_evaluator.hpp').read_text();source=(out/'host_input_evaluator.cpp').read_text()
def once(old,new):
 global source
 if source.count(old)!=1:raise RuntimeError(f'Fragment streaming seam changed: {old[:100]}')
 source=source.replace(old,new)
once('    PathCounts counts{};', '    PathCounts counts{};\n    std::size_t fragment_builds = 0;')
once('      counts        = {};', '      fragment_builds = 0;\n      counts        = {};')
once('    x.counts          = {};', '    x.fragment_builds = 0;\n    x.counts          = {};')
once('''        std::vector<Tensor> right(G.size()), left(G.size());
        for (long b = 0; b < G.size(); ++b)
          if (paths.right[b] && live[paths.right[b]->end_block]) right[b] = fragment(input, backbone, *paths.right[b], false);
        for (long b = 0; b < G.size(); ++b)
          if (paths.left[b] && live[b]) left[b] = fragment(input, backbone, *paths.left[b], true);''', '''        // Cache scope is one index and one call. Evict before building, so only
        // the other slot can overlap a fragment's seven-rank-tensor phase.
        Tensor right, left;
        int right_key=-1, left_key=-1;
        auto ensure_fragment = [&](Tensor &slot, int &key, int root, CorrelatorFragment const &path, bool is_left) {
          if(key==root) return;
          ops->finish();
          slot=Tensor();
          key=-1;
          slot=fragment(input,backbone,path,is_left);
          key=root;
          fragment_builds=checked_add(fragment_builds,1);
          sample_memory();
        };''')
once('''[&](CorrelatorEndpoint const &end, CorrelatorFragment const &, CorrelatorFragment const &) {
                                                       auto const &F''', '''[&](CorrelatorEndpoint const &end, CorrelatorFragment const &rp, CorrelatorFragment const &lp) {
                                                       ensure_fragment(right,right_key,end.right_root,rp,false);
                                                       ensure_fragment(left,left_key,end.left_root,lp,true);
                                                       auto const &F''')
source=source.replace('right[end.right_root]','right').replace('left[end.left_root]','left')
once('''      auto base     = bound();
      auto function = function_bytes(r, model.dimensions());''', '''      auto tile = inventory_memory(r,model.dimensions(),0).maximum_rank_buffer;
      // bound() includes the resident SE output function. Correlators instead
      // need one extra slot during fragment construction: 7 temporary + 1 cached.
      auto base = checked_add(bound()-function_bytes(r,model.dimensions()),tile);''')
once('return checked_add(base, checked_add(function, checked_add(checked_multiply(2, result), checked_multiply(maximum, sizeof(Complex)))));',
 'return checked_add(base, checked_add(checked_multiply(2,result),checked_multiply(maximum,sizeof(Complex))));')
header=header.replace('    PathCounts last_path_counts() const;', '    std::size_t last_fragment_builds() const;\n    PathCounts last_path_counts() const;')
source += '\n#ifdef NDA_HAVE_CUDA\nnamespace triqs_xca::block_sparse::cuda {\nstd::size_t HostInputDiagramEvaluator::last_fragment_builds() const { return impl_->fragment_builds; }\n}\n#endif\n'
header=header.replace('HostInputDiagramEvaluator','StreamFragmentEvaluator')
source=source.replace('HostInputDiagramEvaluator','StreamFragmentEvaluator').replace('"host_input_evaluator.hpp"','"stream_evaluator.hpp"')
(out/'stream_evaluator.hpp').write_text(header);(out/'stream_evaluator.cpp').write_text(source)
