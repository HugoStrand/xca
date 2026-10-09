#!/usr/bin/env python3
"""Start the existing managed daemon and a detached, singleton supervisor."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    path = args.config.resolve()
    cfg = json.loads(path.read_text())
    # Starting an already-running daemon is a no-op; never restart active work.
    with (path.parent / 'launch.log').open('a') as log:
        subprocess.run([cfg['codex_executable'], 'app-server', 'daemon', 'start'],
                       stdout=log, stderr=log, check=True, timeout=60)
    tmux = [cfg['tmux_executable'], '-L', 'xca-cuda-supervisor']
    if subprocess.run(tmux + ['has-session', '-t', '=dense_cuda'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return 0
    command = shlex.join([sys.executable, str(Path(__file__).with_name('supervisor.py')),
                          '--config', str(path)])
    command += ' >> ' + shlex.quote(str(path.parent / 'supervisor.log')) + ' 2>&1'
    subprocess.run(tmux + ['new-session', '-d', '-s', 'dense_cuda', command], check=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
