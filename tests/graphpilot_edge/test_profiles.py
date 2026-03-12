import unittest

from graphpilot_edge.profiles import load_profile_index


class GraphPilotProfilesTest(unittest.TestCase):
    def test_load_profile_index_by_stage_and_backend(self):
        registry = {
            "entries": [
                {
                    "stage_id": "planner.primary",
                    "backend": "cpu",
                    "variant": "gemma3_1b_it",
                    "metrics": {"warm_latency_ms": 420, "peak_memory_bytes": 1048576},
                }
            ]
        }

        index = load_profile_index(registry)
        profile = index[("planner.primary", "cpu")]
        self.assertEqual(profile.warm_latency_ms, 420)
        self.assertEqual(profile.peak_memory_bytes, 1048576)


if __name__ == "__main__":
    unittest.main()
