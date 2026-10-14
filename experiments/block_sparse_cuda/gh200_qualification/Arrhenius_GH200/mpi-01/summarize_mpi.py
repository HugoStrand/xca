#!/usr/bin/env python3
"""Validate MPI ownership records and summarize critical-rank phase times."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    args = manifest['arguments']
    ranks, samples, count = args['ranks'], args['samples'], args['count']
    phases = list(csv.DictReader((folder/'phases.csv').open()))
    jobs = list(csv.DictReader((folder/'jobs.csv').open()))
    policies = ('topology_split', 'flat_cyclic', 'solver_ranges')
    expected = {(s, p, r) for s in range(samples) for p in policies for r in range(ranks)}
    observed = [(int(row['sample']), row['policy'], int(row['rank'])) for row in phases]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Missing or duplicate phase record')
    numeric = [k for k in phases[0] if k not in ('sample', 'policy', 'rank')]
    if any(not math.isfinite(float(row[k])) or float(row[k]) < 0 for row in phases for k in numeric):
        raise ValueError('Invalid/nonfinite phase metric')
    desired_jobs = {(s, p, t, i) for s in range(samples) for p in policies[:2]
                    for t in range(2) for i in range(args['first'], args['first']+count)}
    actual = [(int(row['sample']), row['policy'], int(row['topology']), int(row['index'])) for row in jobs]
    if len(actual) != len(desired_jobs) or set(actual) != desired_jobs:
        raise ValueError('Missing or multiply-owned indexed job')
    if any(not 0 <= int(row['rank']) < ranks or
           not math.isfinite(float(row['api_s'])) or float(row['api_s']) < 0 or
           not math.isfinite(float(row['signal_norm'])) or float(row['signal_norm']) < 0 for row in jobs):
        raise ValueError('Invalid job record')
    summary = []
    for policy in policies:
        measurements = []
        for sample in range(samples):
            group = [row for row in phases if row['policy'] == policy and int(row['sample']) == sample]
            local = [float(row['evaluation_s']) for row in group]
            if sum(int(row['indexed_jobs']) for row in group) != 2*count:
                raise ValueError('Incorrect distributed work count')
            values = dict(critical_total_s=max(float(row['total_s']) for row in group),
                          evaluation_imbalance=max(local)/statistics.mean(local),
                          payload_bytes_per_rank=int(group[0]['payload_bytes']),
                          collectives_per_rank=int(group[0]['collectives']))
            for phase in ('evaluation', 'packing', 'error_coordination', 'arrival_wait', 'reduction', 'accumulation'):
                values[f'max_{phase}_s'] = max(float(row[f'{phase}_s']) for row in group)
            measurements.append(values)
        item = dict(policy=policy, samples=samples, ranks=ranks)
        for key in measurements[0]:
            item[key] = statistics.median(row[key] for row in measurements)
        item['min_critical_total_s'] = min(row['critical_total_s'] for row in measurements)
        item['max_critical_total_s'] = max(row['critical_total_s'] for row in measurements)
        summary.append(item)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.folder)
    with args.output.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
