"""Verify the actual process lock and launcher without starting model turns."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch


class ProcessTests(unittest.TestCase):
    def test_second_process_cannot_acquire_runtime_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = root / 'config.json'
            cfg.write_text('{}')
            with (root / 'supervisor.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                r = subprocess.run([sys.executable, str(Path(launch.__file__).with_name('supervisor.py')),
                                    '--config', str(cfg), '--once'], capture_output=True, text=True, timeout=5)
                self.assertEqual(r.returncode, 2)
                self.assertIn('already owns', r.stderr)

    def test_existing_tmux_session_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / 'config.json'
            cfg.write_text(json.dumps({'codex_executable': '/test/codex', 'tmux_executable': '/test/tmux'}))
            with patch.object(sys, 'argv', ['launch', '--config', str(cfg)]), patch('launch.subprocess.run') as run:
                run.return_value.returncode = 0
                self.assertEqual(launch.main(), 0)
                self.assertEqual(len(run.call_args_list), 2)
                self.assertEqual(run.call_args_list[0].args[0], ['/test/codex', 'app-server', 'daemon', 'start'])
                self.assertNotIn('new-session', run.call_args_list[1].args[0])

    def test_detached_launch_quotes_paths(self):
        with tempfile.TemporaryDirectory(prefix='supervisor space ') as d:
            cfg = Path(d) / 'config.json'
            cfg.write_text(json.dumps({'codex_executable': '/test/codex', 'tmux_executable': '/test/tmux'}))
            with patch.object(sys, 'argv', ['launch', '--config', str(cfg)]), patch('launch.subprocess.run') as run:
                run.return_value.returncode = 1
                self.assertEqual(launch.main(), 0)
                self.assertEqual(len(run.call_args_list), 3)
                self.assertIn('new-session', run.call_args_list[-1].args[0])
                self.assertIn("'" + str(cfg) + "'", run.call_args_list[-1].args[0][-1])


if __name__ == '__main__': unittest.main()
