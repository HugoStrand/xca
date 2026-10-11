#!/usr/bin/env python3
"""Fail if an experiment changes any frozen production evaluator source."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
manifest=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
changed=[p for p,h in manifest['sha256'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
if changed: raise SystemExit('Frozen baseline changed: '+', '.join(changed))
print(f"Baseline {manifest['baseline_revision']}: {len(manifest['sha256'])} source files unchanged")
