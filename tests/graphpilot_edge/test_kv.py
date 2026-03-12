import unittest

from graphpilot_edge.kv import DegradationAction, KvSession, choose_degradation_action, can_admit_session


class GraphPilotKvTest(unittest.TestCase):
    def test_can_admit_session_accounts_for_margin(self):
        sessions = [KvSession("s1", kv_bytes=200)]
        self.assertTrue(can_admit_session(sessions, workbuf_bytes=100, other_live_bytes=50, budget_bytes=400, margin_bytes=25))
        self.assertFalse(can_admit_session(sessions, workbuf_bytes=150, other_live_bytes=75, budget_bytes=400, margin_bytes=25))

    def test_choose_degradation_action_minimizes_quality_and_latency_harm(self):
        actions = [
            DegradationAction("reduce_retrieval", freed_bytes=128, added_latency_ms=10, quality_loss=0.3),
            DegradationAction("reduce_context", freed_bytes=64, added_latency_ms=1, quality_loss=0.05),
        ]
        selected = choose_degradation_action(actions, latency_weight=0.1)
        self.assertEqual(selected.action_id, "reduce_context")


if __name__ == "__main__":
    unittest.main()
