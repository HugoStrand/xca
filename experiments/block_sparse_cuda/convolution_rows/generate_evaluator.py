"""Generate a private complete evaluator; fail closed if production seams change."""
from pathlib import Path
import sys
root, out = map(Path, sys.argv[1:])
cuda = root / 'experiments/block_sparse_cuda/reference_snapshot'
header = (cuda / 'diagram_evaluator.hpp').read_text()
source = (cuda / 'diagram_evaluator.cpp').read_text()

def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Row evaluator seam changed: {old[:100]}')
    return text.replace(old, new)

header = once(header, 'NcaBackend nca_backend = NcaBackend::Reference);',
              'NcaBackend nca_backend = NcaBackend::Reference, long row_cap = 8);')
header = once(header, '    PathCounts last_path_counts() const;', '    void sample_row_allocations(bool enabled);\n    PathCounts last_path_counts() const;')
header = header.replace('DiagramEvaluator', 'RowDiagramEvaluator').replace('"dlr_ops.hpp"', '"row_dlr.hpp"')
source = once(source, '    long r;', '    long row_cap;\n    long r;')
source = once(source, 'std::size_t host_limit, NcaBackend nca_selected)',
              'std::size_t host_limit, NcaBackend nca_selected, long selected_rows)')
source = once(source, '         r(mesh.dlr_it().rank()),', '         row_cap(selected_rows),\n         r(mesh.dlr_it().rank()),')
source = once(source, '      if (placement !=', '      if (row_cap <= 0) throw std::invalid_argument("Invalid convolution row cap");\n      if (placement !=')
source = once(source, '      dlr_bound    = checked_add(tables, checked_multiply(5, tile));', '''      auto maximum = *std::max_element(model.dimensions().begin(), model.dimensions().end());
      // Each factor dimension is <= maximum. RHS <= tile and is freed before
      // row convolution. Full transformed right + two B*K + three B*N tensors.
      auto strip = tensor_bytes(r, std::min(row_cap, maximum), maximum);
      dlr_bound = checked_add(tables, checked_add(tile, checked_multiply(5, strip)));''')
source = source.replace('dlr->convolve(beta, gc, tc, out);', 'dlr->convolve_rows(beta, gc, tc, out, row_cap);')
source = source.replace('dlr->convolve(beta, tc, gc, out);', 'dlr->convolve_rows(beta, tc, gc, out, row_cap);')
assert source.count('dlr->convolve_rows(') == 3
source = once(source, 'std::size_t host_output_budget, NcaBackend nca_backend)',
              'std::size_t host_output_budget, NcaBackend nca_backend, long row_cap)')
source = once(source, 'output_placement, host_output_budget, nca_backend))',
              'output_placement, host_output_budget, nca_backend, row_cap))')
source += '''\n#ifdef NDA_HAVE_CUDA\nnamespace triqs_xca::block_sparse::cuda {\nvoid DiagramEvaluator::sample_row_allocations(bool enabled) {\n  impl_->ops->require_current();\n  if (enabled) impl_->dlr->set_allocation_observer([this] { impl_->sample_memory(); });\n  else impl_->dlr->set_allocation_observer({});\n}\n}\n#endif\n'''
source = source.replace('DiagramEvaluator', 'RowDiagramEvaluator').replace('DlrOps', 'RowDlrOps')
source = source.replace('"diagram_evaluator.hpp"', '"row_evaluator.hpp"')
(out / 'row_evaluator.hpp').write_text(header)
(out / 'row_evaluator.cpp').write_text(source)
