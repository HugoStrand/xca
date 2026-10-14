#!/usr/bin/env python3
"""Summarize structural reference work; operand footprints are not HBM traffic."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups, inputs = defaultdict(lambda: [0, 0, 0, 0]), []
    names = set()
    for path in args.traces:
        name = path.stem
        if name in names:
            raise ValueError('Trace basenames must be unique')
        names.add(name)
        count = 0
        with path.open() as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                count += 1
                if count > 200000:
                    raise ValueError('Trace exceeds declared row cap')
                values = [int(row[k]) for k in ('flops_model', 'dense_equivalent_flops', 'operand_footprint_bytes')]
                if min(values) < 0 or any(int(row[k]) < 1 for k in ('M', 'N', 'K', 'batches', 'rank')):
                    raise ValueError('Invalid work extent/count')
                if values[0] > values[1]:
                    raise ValueError('Sparse work exceeds dense equivalent')
                scope = 'potential_after_cutoff' if row['kind'] == 'possible_output_sum' else 'structural_model'
                key = name, scope, row['kind']
                groups[key] = [a+b for a, b in zip(groups[key], [1, *values])]
        inputs.append(dict(path=str(path.resolve()), rows=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for (name, scope, kind), values in groups.items():
        totals[name, scope] = [a+b for a, b in zip(totals[name, scope], values)]
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ['case', 'scope', 'kind', 'events', 'flops_model', 'dense_equivalent_flops', 'operand_footprint_bytes',
              'flops_fraction', 'footprint_fraction']
    with (args.output/'weighted.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for (name, scope, kind), values in sorted(groups.items()):
            total = totals[name, scope]
            writer.writerow(dict(zip(fields, [name, scope, kind, *values,
                values[1]/total[1] if total[1] else 0., values[3]/total[3] if total[3] else 0.])))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        schema='block-sparse-work-v1', inputs=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All-live structural self-energy candidates, reference schedule; not numerical survivors, traffic or timings',
        exclusions='Allocation, transfers/packing, zero fill, scans, pivoting/division, caches and opaque library work; LU is leading cost',
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
