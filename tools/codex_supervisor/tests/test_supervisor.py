"""Quota recovery tests use a fake server; no inference usage is consumed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervisor import Supervisor, quota_wait, save


class FakeRpc:
    def __init__(self):
        self.goal = {'status': 'active', 'objective': 'port'}
        self.thread = {'cwd': '/tmp/xca', 'status': {'type': 'idle'}, 'updatedAt': 1}
        self.turn = {'id': 'old', 'status': 'completed', 'error': None}
        self.limits = {'ordinaryUsageAllowed': True, 'rateLimits': {'primary': {'usedPercent': 25}}}
        self.calls = []
        self.hook = None

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        if self.hook:
            result = self.hook(method, params)
            if result is not None:
                return result
        if method == 'thread/goal/get': return {'goal': copy.deepcopy(self.goal)}
        if method == 'thread/read': return {'thread': copy.deepcopy(self.thread)}
        if method == 'thread/turns/list': return {'data': [copy.deepcopy(self.turn)]}
        if method == 'account/rateLimits/read': return copy.deepcopy(self.limits)
        if method == 'thread/resume':
            self.thread['status']['type'] = 'idle'
            return {'thread': copy.deepcopy(self.thread)}
        if method == 'thread/goal/set':
            self.goal['status'] = params['status']
            return {'goal': copy.deepcopy(self.goal)}
        if method == 'turn/start':
            self.turn = {'id': 'new', 'status': 'inProgress'}
            self.thread['status']['type'] = 'active'
            return {'turn': self.turn}
        raise AssertionError(method)

    def mutations(self):
        return [(m, p) for m, p in self.calls if m in {'thread/goal/set', 'turn/start', 'thread/resume'}]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = {'repo': '/tmp/xca', 'branch': 'dev_cuda', 'thread_id': 'test',
                       'objective': 'port', 'model': 'gpt-6.1-sol', 'resume_prompt': 'continue',
                       'idle_grace': 120}
        self.state = {}
        self.saved = []
        self.supervisor = Supervisor(self.config, self.state,
                                     lambda s: self.saved.append(copy.deepcopy(s)), lambda: 'dev_cuda')
        self.rpc = FakeRpc()

    def tick(self, t=0, **kwargs):
        with patch('builtins.print'):
            return self.supervisor.tick(self.rpc, now=t, **kwargs)[0]

    def test_idle_requires_grace_then_starts_once(self):
        self.assertEqual(self.tick(), 'idle_grace')
        self.assertEqual(self.tick(119), 'idle_grace')
        self.assertEqual(self.tick(120), 'resumed')
        self.assertEqual(self.tick(150), 'thread_active')
        starts = [(m, p) for m, p in self.rpc.calls if m == 'turn/start']
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][1]['effort'], 'medium')
        self.assertEqual(starts[0][1]['serviceTier'], 'default')
        self.assertNotIn('sandboxPolicy', starts[0][1])
        self.assertNotIn('approvalPolicy', starts[0][1])
        self.assertTrue(any(s.get('pending') for s in self.saved))

    def test_quota_reset_needs_explicit_backend_recovery(self):
        self.rpc.goal['status'] = 'usageLimited'
        self.rpc.turn.update(status='failed', error={'codexErrorInfo': 'usageLimitExceeded'})
        self.rpc.limits = {'ordinaryUsageAllowed': False, 'rateLimits': {
            'primary': {'usedPercent': 100, 'resetsAt': 50}}}
        self.assertEqual(self.tick(), 'waiting_quota')
        self.assertEqual(self.tick(60), 'waiting_quota')
        self.assertFalse(self.rpc.mutations())
        self.rpc.limits['ordinaryUsageAllowed'] = True
        self.rpc.limits['rateLimits']['primary']['usedPercent'] = 0
        self.assertEqual(self.tick(70), 'idle_grace')
        self.assertEqual(self.tick(190), 'resumed')
        mutations = self.rpc.mutations()
        self.assertEqual([m for m, _ in mutations], ['thread/goal/set', 'turn/start'])
        self.assertEqual(mutations[0][1]['origin'], 'automatic')

    def test_configured_model_effort_is_preserved(self):
        self.config.update(model='gpt-6-astra', effort='high')
        self.tick()
        self.assertEqual(self.tick(120), 'resumed')
        start = next(p for m, p in self.rpc.calls if m == 'turn/start')
        self.assertEqual((start['model'], start['effort']), ('gpt-6-astra', 'high'))

    def test_system_error_quota_reset_recovers_for_both_goal_states(self):
        for status in ('active', 'usageLimited'):
            with self.subTest(status=status):
                self.setUp()
                self.rpc.goal['status'] = status
                self.rpc.thread['status']['type'] = 'systemError'
                self.rpc.turn.update(status='failed', error={'codexErrorInfo': 'usageLimitExceeded'})
                self.rpc.limits['ordinaryUsageAllowed'] = False
                self.assertEqual(self.tick(), 'waiting_quota')
                self.assertEqual(self.tick(10000), 'waiting_quota')
                self.assertFalse(self.rpc.mutations())
                self.rpc.limits['ordinaryUsageAllowed'] = True
                self.assertEqual(self.tick(10030), 'idle_grace')
                self.assertEqual(self.tick(10150), 'resumed')
                self.assertEqual(self.tick(10180), 'thread_active')
                self.assertEqual(len([m for m, _ in self.rpc.calls if m == 'turn/start']), 1)
                self.assertNotIn('thread/resume', [m for m, _ in self.rpc.calls])

    def test_system_error_without_terminal_quota_failure_stays_blocked(self):
        self.rpc.thread['status']['type'] = 'systemError'
        for status, error in [('completed', None), ('interrupted', {'codexErrorInfo': 'usageLimitExceeded'}),
                              ('failed', {'codexErrorInfo': 'sandboxError'}), ('failed', None),
                              ('failed', {'codexErrorInfo': {'other': 'unknown'}})]:
            with self.subTest(status=status, error=error):
                self.rpc.turn.update(status=status, error=error)
                self.assertEqual(self.tick(10000), 'waiting_thread_error')
        self.assertFalse(self.rpc.mutations())

    def test_system_error_rechecks_latest_failure_before_start(self):
        self.rpc.thread['status']['type'] = 'systemError'
        self.rpc.turn.update(status='failed', error={'codexErrorInfo': 'rateLimitExceeded'})
        self.tick()
        reads = 0
        def hook(method, params):
            nonlocal reads
            if method == 'thread/turns/list':
                reads += 1
                if reads == 2:
                    return {'data': [{'id': 'newer', 'status': 'failed',
                                      'error': {'codexErrorInfo': 'sandboxError'}}]}
        self.rpc.hook = hook
        self.assertEqual(self.tick(120), 'thread_became_active')
        self.assertFalse(self.rpc.mutations())

    def test_permission_revoked_during_final_checks_does_not_resume(self):
        self.tick()
        reads = 0
        def hook(method, params):
            nonlocal reads
            if method == 'account/rateLimits/read':
                reads += 1
                if reads == 2: return {'ordinaryUsageAllowed': False}
        self.rpc.hook = hook
        self.assertEqual(self.tick(120), 'waiting_quota')
        self.assertFalse(self.rpc.mutations())

    def test_quota_wait_after_idle_restarts_grace(self):
        self.assertEqual(self.tick(), 'idle_grace')
        self.rpc.limits['ordinaryUsageAllowed'] = False
        self.assertEqual(self.tick(30), 'waiting_quota')
        self.rpc.limits['ordinaryUsageAllowed'] = True
        self.assertEqual(self.tick(60), 'idle_grace')
        self.assertEqual(self.tick(180), 'resumed')

    def test_weekly_window_still_blocks(self):
        self.rpc.limits = {'ordinaryUsageAllowed': False, 'rateLimits': {
            'primary': {'usedPercent': 0, 'resetsAt': 50},
            'secondary': {'usedPercent': 100, 'resetsAt': 100000}}}
        self.assertEqual(self.tick(60), 'waiting_quota')
        self.assertFalse(self.rpc.mutations())

    def test_unknown_permission_and_inconsistent_snapshot_block(self):
        self.rpc.limits['ordinaryUsageAllowed'] = None
        self.assertEqual(self.tick(), 'usage_unknown')
        self.rpc.limits['ordinaryUsageAllowed'] = True
        self.rpc.limits['rateLimits']['primary']['usedPercent'] = 100
        self.assertEqual(self.tick(200), 'inconsistent_usage_snapshot')
        self.assertFalse(self.rpc.mutations())

    def test_workspace_credits_are_not_treated_as_window_reset(self):
        self.rpc.limits['rateLimits']['rateLimitReachedType'] = 'workspace_owner_credits_depleted'
        self.assertEqual(self.tick(999999), 'waiting_workspace_allowance')
        self.assertFalse(self.rpc.mutations())

    def test_active_and_approval_wait_never_resume(self):
        self.rpc.thread['status'] = {'type': 'active', 'activeFlags': ['waitingOnApproval']}
        self.assertEqual(self.tick(99999), 'thread_active')
        self.assertFalse(self.rpc.mutations())
        self.assertNotIn('account/rateLimits/read', [m for m, _ in self.rpc.calls])

    def test_user_pause_block_budget_and_completion_are_respected(self):
        for status in ['paused', 'blocked', 'budgetLimited', 'complete']:
            with self.subTest(status=status):
                self.rpc.goal['status'] = status
                self.assertEqual(self.tick(99999), 'goal_' + status)
        self.assertFalse(self.rpc.mutations())

    def test_branch_and_objective_guard(self):
        self.supervisor.branch_reader = lambda: 'main'
        self.assertEqual(self.tick(), 'waiting_branch')
        self.supervisor.branch_reader = lambda: 'dev_cuda'
        self.rpc.goal['objective'] = 'unrelated'
        self.assertEqual(self.tick(), 'waiting_goal_identity')
        self.assertFalse(self.rpc.mutations())

    def test_interrupt_and_other_failures_require_manual_recovery(self):
        for status, error in [('interrupted', None), ('failed', {'codexErrorInfo': 'sandboxError'}),
                              ('failed', {'codexErrorInfo': 'sessionBudgetExceeded'})]:
            self.rpc.turn.update(status=status, error=error)
            self.assertEqual(self.tick(99999), 'waiting_interruption_or_failure')
        self.assertFalse(self.rpc.mutations())

    def test_user_pauses_between_checks(self):
        self.tick()
        count = 0
        def hook(method, params):
            nonlocal count
            if method == 'thread/goal/get':
                count += 1
                if count == 2: return {'goal': {'status': 'paused', 'objective': 'port'}}
        self.rpc.hook = hook
        self.assertEqual(self.tick(120), 'goal_changed')
        self.assertFalse(self.rpc.mutations())

    def test_active_race_does_not_start(self):
        self.tick()
        reads = 0
        def hook(method, params):
            nonlocal reads
            if method == 'thread/read':
                reads += 1
                if reads == 2: return {'thread': {'status': {'type': 'active'}}}
        self.rpc.hook = hook
        self.assertEqual(self.tick(120), 'thread_became_active')
        self.assertFalse(self.rpc.mutations())

    def test_dry_run_and_unloaded_thread(self):
        self.rpc.thread['status']['type'] = 'notLoaded'
        self.tick()
        self.assertEqual(self.tick(120, dry_run=True), 'would_resume')
        self.assertFalse(self.rpc.mutations())
        self.assertEqual(self.tick(121), 'resumed')
        self.assertEqual(self.rpc.mutations()[0][0], 'thread/resume')

    def test_lost_start_reply_survives_restart_without_duplicate(self):
        self.tick()
        def hook(method, params):
            if method == 'turn/start': raise TimeoutError('lost response')
        self.rpc.hook = hook
        with self.assertRaises(TimeoutError): self.tick(120)
        self.assertTrue(self.state['pending'])
        self.supervisor = Supervisor(self.config, self.state, lambda s: None, lambda: 'dev_cuda')
        self.rpc.hook = None
        self.assertEqual(self.tick(150), 'uncertain_start')
        self.assertEqual(self.tick(180), 'halted_manual_review')
        self.assertEqual(len([m for m, _ in self.rpc.calls if m == 'turn/start']), 1)

    def test_lost_reply_with_visible_new_turn_can_recover(self):
        self.tick()
        def hook(method, params):
            if method == 'turn/start':
                self.rpc.turn = {'id': 'new', 'status': 'completed'}
                raise TimeoutError('lost response')
        self.rpc.hook = hook
        with self.assertRaises(TimeoutError): self.tick(120)
        self.rpc.hook = None
        self.assertEqual(self.tick(150), 'idle_grace')
        self.assertNotIn('pending', self.state)

    def test_state_atomic_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'
            save(p, {'pending': True})
            self.assertEqual(json.loads(p.read_text()), {'pending': True})
            self.assertFalse(p.with_suffix('.tmp').exists())


if __name__ == '__main__': unittest.main()
