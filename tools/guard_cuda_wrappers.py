#!/usr/bin/env python3
"""Keep clair-generated CUDA bindings usable with XCA_CUDA_SMALL=OFF.

clair emits declarations for the complete schema. Guard the optional type's
class definition and registration without changing dispatch or type names.
"""

import re
import sys
from pathlib import Path


def guard_wrappers(path):
    text = path.read_text()
    match = re.search(r'using (_c2py_cls_\d+)\s*= triqs_xca::dense::cuda_small::DiagramEvaluator;', text)
    if match is None:
        raise RuntimeError('Expected the generated small CUDA evaluator class')
    alias = match.group(1)
    start = text.rfind('// --------- class ', 0, match.start())
    ends = [position for marker in ('// --------- class ', '// ==================== module functions')
            if (position := text.find(marker, match.end())) != -1]
    if start == -1 or not ends:
        raise RuntimeError('Unrecognized generated class boundaries')
    end = min(ends)
    text = text[:start] + '#ifdef XCA_HAVE_CUDA_SMALL\n' + text[start:end] + '#endif\n\n' + text[end:]
    for pattern in (rf'^.*PyType_Ready\(&c2py::wrap_pytype<{alias}>\).*$',
                    rf'^.*_add_type\({alias}, "DenseCudaSmallDiagramEvaluator"\);.*$'):
        text, count = re.subn(pattern, lambda m: '#ifdef XCA_HAVE_CUDA_SMALL\n' + m.group() + '\n#endif', text, flags=re.M)
        if count != 1:
            raise RuntimeError('Expected one generated registration for ' + alias)
    path.write_text(text)
    header = path.with_suffix('.hxx')
    text = header.read_text()
    start = text.index('template <> constexpr bool c2py::is_wrapped<triqs_xca::dense::cuda_small::DiagramEvaluator>')
    end = text.index(';', text.index('"triqs_xca._triqs_xca_cuda.DenseCudaSmallDiagramEvaluator"', start)) + 1
    header.write_text(text[:start] + '#ifdef XCA_HAVE_CUDA_SMALL\n' + text[start:end] + '\n#endif' + text[end:])


if __name__ == '__main__':
    guard_wrappers(Path(sys.argv[1]))
