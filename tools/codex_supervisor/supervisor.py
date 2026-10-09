#!/usr/bin/env python3
"""Resume one authorized Codex goal through its existing shared app-server."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from unix_rpc import UnixRpc

QUOTA_ERRORS = {'usageLimitExceeded', 'rateLimitExceeded'}
TERMINAL_GOALS = {'paused', 'blocked', 'budgetLimited', 'complete'}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def quota_wait(limits, now, model):
    """Reset timers select polling intervals; only the backend permits recovery."""
    buckets = [limits.get('rateLimits') or {}]
    for name, bucket in (limits.get('rateLimitsByLimitId') or {}).items():
        if name in {'codex', model} or bucket.get('normalModelSlug') == model:
            buckets.append(bucket)
    for b in buckets:
        reached = b.get('rateLimitReachedType')
        if b.get('spendControlReached') or (reached and reached != 'rate_limit_reached'):
            return 'waiting_workspace_allowance', 900
    if limits.get('ordinaryUsageAllowed') is not True:
        resets = [w['resetsAt'] for b in buckets for k in ('primary', 'secondary')
                  if (w := b.get(k)) and w.get('usedPercent', 0) >= 100
                  and isinstance(w.get('resetsAt'), (int, float)) and w['resetsAt'] > now]
        delay = min(900, max(60, max(resets, default=now + 300) - now + 30))
        return 'waiting_quota' if limits.get('ordinaryUsageAllowed') is False else 'usage_unknown', delay
    if any(b.get('rateLimitReachedType') or
           any((b.get(k) or {}).get('usedPercent', 0) >= 100 for k in ('primary', 'secondary'))
           for b in buckets):
        return 'inconsistent_usage_snapshot', 300
    return None, 30


class Supervisor:
    def __init__(self, config, state, persist, branch_reader=None):
        self.config, self.state, self.persist = config, state, persist
        self.branch_reader = branch_reader or self._branch

    def _branch(self):
        return subprocess.check_output(['git', '-C', self.config['repo'], 'branch', '--show-current'],
                                       text=True, timeout=10).strip()

    def record(self, action, now, **fields):
        previous = self.state.get('action')
        self.state.update(action=action, checked_at=now, **fields)
        self.persist(self.state)
        if previous != action:
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)), action, flush=True)
        return action

    def tick(self, rpc, now=None, dry_run=False):
        now = time.time() if now is None else now
        cfg, state = self.config, self.state
        if state.get('halted'):
            return self.record('halted_manual_review', now), 900
        if self.branch_reader() != cfg['branch']:
            return self.record('waiting_branch', now), 60
        tid = cfg['thread_id']
        goal = rpc.call('thread/goal/get', {'threadId': tid}).get('goal')
        if not goal:
            return self.record('no_goal', now), 60
        if goal.get('objective') != cfg['objective']:
            return self.record('waiting_goal_identity', now), 60
        status = goal.get('status')
        if status in TERMINAL_GOALS:
            state.pop('idle_since', None)
            return self.record('goal_' + status, now), 300
        if status not in {'active', 'usageLimited'}:
            return self.record('unknown_goal_status', now), 300
        thread = rpc.call('thread/read', {'threadId': tid, 'includeTurns': False})['thread']
        if Path(thread['cwd']).resolve() != Path(cfg['repo']).resolve():
            return self.record('waiting_workspace_identity', now), 300
        runtime = thread['status']['type']
        if runtime == 'active':
            state.pop('idle_since', None)
            return self.record('thread_active', now), 30
        if runtime not in {'idle', 'notLoaded'}:
            return self.record('waiting_thread_error', now), 300
        turns = rpc.call('thread/turns/list', {'threadId': tid, 'limit': 1,
                         'sortDirection': 'desc', 'itemsView': 'summary'})['data']
        last = turns[0] if turns else {}
        error = (last.get('error') or {}).get('codexErrorInfo')
        if not isinstance(error, str): error = None
        if last.get('status') == 'inProgress':
            return self.record('waiting_unfinished_turn', now), 60
        if state.get('pending'):
            if last.get('id') == state.get('before_turn'):
                return self.record('uncertain_start', now, halted=True), 900
            state.pop('pending', None)
        if last.get('status') in {'failed', 'interrupted'} and error not in QUOTA_ERRORS:
            return self.record('waiting_interruption_or_failure', now), 300
        wait, delay = quota_wait(rpc.call('account/rateLimits/read'), now, cfg['model'])
        if wait:
            state.pop('idle_since', None)
            return self.record(wait, now), delay
        key = [status, last.get('id'), thread.get('updatedAt')]
        if state.get('idle_key') != key or 'idle_since' not in state:
            state.update(idle_key=key, idle_since=now)
        if now - state.get('idle_since', now) < cfg.get('idle_grace', 120):
            return self.record('idle_grace', now), 30
        if dry_run:
            return self.record('would_resume', now), 30
        if rpc.call('thread/goal/get', {'threadId': tid}).get('goal') != goal:
            state.pop('idle_since', None)
            return self.record('goal_changed', now), 30
        if runtime == 'notLoaded':
            rpc.call('thread/resume', {'threadId': tid, 'excludeTurns': True})
        if rpc.call('thread/read', {'threadId': tid, 'includeTurns': False})['thread']['status']['type'] != 'idle':
            state.pop('idle_since', None)
            return self.record('thread_became_active', now), 30
        if status == 'usageLimited':
            # Automatic provenance preserves user pauses and requires backend recovery.
            rpc.call('thread/goal/set', {'threadId': tid, 'status': 'active', 'origin': 'automatic'})
            if rpc.call('thread/read', {'threadId': tid, 'includeTurns': False})['thread']['status']['type'] != 'idle':
                return self.record('thread_became_active', now), 30
        # Journal before starting; a lost response is reconciled rather than retried.
        state.update(pending=True, before_turn=last.get('id'), request_id=str(uuid.uuid4()))
        self.record('starting_turn', now)
        result = rpc.call('turn/start', {
            'threadId': tid, 'input': [{'type': 'text', 'text': cfg['resume_prompt']}],
            'model': cfg['model'], 'effort': 'medium', 'serviceTier': 'default',
            'clientUserMessageId': state['request_id']})
        state.pop('pending', None)
        state.pop('idle_since', None)
        return self.record('resumed', now, last_started_turn=result['turn']['id']), 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Never mutate goal/thread')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = args.config.parent
    os.umask(0o077)
    with (runtime / 'supervisor.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.exit(2, 'A supervisor already owns this runtime directory\n')
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + '\n'); lock.flush()
        path = runtime / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        binding = {k: config[k] for k in ('thread_id', 'repo', 'branch', 'objective')}
        if state.get('binding') not in (None, binding):
            parser.exit(2, 'State belongs to another goal; use a new runtime directory\n')
        state['binding'] = binding
        supervisor = Supervisor(config, state, lambda v: save(path, v))
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        failures = 0
        while not stopping:
            try:
                if (runtime / 'PAUSE').exists():
                    action, delay = supervisor.record('local_pause', time.time()), 30
                else:
                    with UnixRpc(config['socket'], timeout=30) as rpc:
                        action, delay = supervisor.tick(rpc, dry_run=args.dry_run)
                failures = 0
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failures += 1
                action = supervisor.record('connection_or_protocol_error', time.time(), error_type=type(exc).__name__)
                delay = min(900, 30 * 2 ** min(failures, 5))
            if args.once: return 1 if action == 'connection_or_protocol_error' else 0
            if action in {'goal_complete', 'no_goal', 'waiting_goal_identity', 'halted_manual_review', 'uncertain_start'}:
                return 0
            deadline = time.monotonic() + delay
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    return 0


if __name__ == '__main__': raise SystemExit(main())
