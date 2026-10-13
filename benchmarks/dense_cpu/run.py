#!/usr/bin/env python3
"""Record CPU-only evaluator measurements using the shared workload/CSV schema."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dense_cuda_small'))
from run import ROOT, main

if __name__ == '__main__':
    sys.exit(main(default_binary=ROOT/'cbuild/benchmarks/dense_cpu/dense_cpu_benchmark',
                  default_extra=('--backend', 'cpu')))
