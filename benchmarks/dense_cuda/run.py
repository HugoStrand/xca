#!/usr/bin/env python3
"""Record dense_cuda measurements using the shared dense workload recorder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dense_common'))
from runner import ROOT, main

if __name__ == '__main__':
    sys.exit(main(default_binary=ROOT/'cbuild-cuda/benchmarks/dense_cuda/dense_cuda_fixture_benchmark',
                  default_extra=(),
                  source_directories=(ROOT/'benchmarks/dense_cuda', ROOT/'c++/triqs_xca/dense/cuda',)))
