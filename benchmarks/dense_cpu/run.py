#!/usr/bin/env python3
"""Record dense_cpu measurements using the shared dense workload recorder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dense_common'))
from runner import ROOT, main

if __name__ == '__main__':
    sys.exit(main(default_binary=ROOT/'cbuild/benchmarks/dense_cpu/dense_cpu_benchmark',
                  default_extra=('--backend', 'cpu'),
                  source_directories=(ROOT/'benchmarks/dense_cpu',)))
