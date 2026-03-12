import unittest

from graphpilot_edge.hardware_topology import DEFAULT_HARDWARE_TOPOLOGY_PATH, load_hardware_topology


class HardwareTopologyTest(unittest.TestCase):
    def test_default_topology_loads_simulator_and_metadata(self):
        loaded = load_hardware_topology(DEFAULT_HARDWARE_TOPOLOGY_PATH)

        self.assertEqual(loaded.topology_id, "sm8750_surrogate_v1")
        self.assertIn("cpu0", loaded.simulator.resource_ids())
        self.assertIn("gpu0", loaded.simulator.resource_ids())
        self.assertIn("npu0", loaded.simulator.resource_ids())
        self.assertEqual(loaded.simulator.backend_label("npu0"), "npu")
        self.assertTrue(loaded.public_anchors)
        self.assertTrue(loaded.surrogate_parameter_notes)


if __name__ == "__main__":
    unittest.main()
