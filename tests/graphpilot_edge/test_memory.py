import unittest

from graphpilot_edge.memory import BufferRequest, allocate_intervals


class GraphPilotMemoryTest(unittest.TestCase):
    def test_allocate_intervals_reuses_non_overlapping_buffers(self):
        requests = [
            BufferRequest("a", start_ms=0, end_ms=5, size_bytes=64, memory_type="cpu"),
            BufferRequest("b", start_ms=5, end_ms=8, size_bytes=32, memory_type="cpu"),
            BufferRequest("c", start_ms=2, end_ms=7, size_bytes=16, memory_type="cpu"),
        ]

        result = allocate_intervals(requests)
        self.assertEqual(result.assignment["a"], result.assignment["b"])
        self.assertNotEqual(result.assignment["a"], result.assignment["c"])
        self.assertEqual(result.peak_bytes, 80)


if __name__ == "__main__":
    unittest.main()
