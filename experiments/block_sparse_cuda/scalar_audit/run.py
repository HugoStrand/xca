#!/usr/bin/env python3
"""Bounded correctness audit; reported durations are not performance evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
binary = a.binary.resolve()
manifest = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), cases=[], timeout_seconds=120)
for name, window, eps, order, pattern in [('smooth-r23', 20, 1e-10, 2, 'smooth'), ('oscillating', 60, 1e-12, 2, 'oscillating'),
                                         ('complex', 100, 1e-14, 2, 'complex'), ('order6', 20, 1e-10, 6, 'smooth')]:
    for repair in [False, True]:
        label = name + ('-repair' if repair else '-band')
        cmd = [str(binary), '--lambda', str(window), '--eps', str(eps), '--order', str(order), '--pattern', pattern]
        if repair:
            cmd.append('--repair')
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr, code = error.stdout or b'', error.stderr or b'', None
            stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
        (a.output / (label + '.csv')).write_text(stdout)
        (a.output / (label + '.log')).write_text(stderr)
        manifest['cases'].append(dict(name=label, command=cmd, returncode=code))
        (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if code != 0:
            raise SystemExit('Stopped on failure: ' + label)
        rows = list(csv.DictReader(stdout.splitlines()))
        assert len(rows) == 59 and all(int(r['lanes']) == 1 and int(r['nonfinite']) == 0 for r in rows)
        print(label, 'rank', rows[0]['rank'], 'outside', sum(int(r['outside_band']) for r in rows),
              'count failures', sum(r['reference_count'] != r['candidate_count'] for r in rows), flush=True)
