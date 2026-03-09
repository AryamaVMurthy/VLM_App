import importlib.util
import pathlib
import sys
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "fastvlm_budget_controller.py"
    spec = importlib.util.spec_from_file_location("fastvlm_budget_controller", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BudgetControllerTest(unittest.TestCase):
    def test_choose_visual_token_budget_lowers_budget_under_queue_pressure(self):
        module = load_module()

        decision = module.choose_visual_token_budget(
            [96, 160, 256],
            queue_depth=5,
            queued_image_count=5,
            recent_prefill_ms=None,
            recent_decode_ms=None,
            answer_mode="none",
        )

        self.assertEqual(decision.budget, 160)
        self.assertIn("queue_pressure", decision.reason_codes)

    def test_choose_visual_token_budget_matches_decode_window(self):
        module = load_module()

        decision = module.choose_visual_token_budget(
            [96, 160, 256],
            queue_depth=0,
            queued_image_count=1,
            recent_prefill_ms=320.0,
            recent_decode_ms=200.0,
            answer_mode="short",
        )

        self.assertEqual(decision.budget, 96)
        self.assertIn("prefill_above_decode_window", decision.reason_codes)
        self.assertIn("short_answer_mode", decision.reason_codes)

    def test_choose_visual_token_budget_rejects_invalid_buckets(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "positive"):
            module.choose_visual_token_budget(
                [0, 96],
                queue_depth=0,
                queued_image_count=0,
                recent_prefill_ms=None,
                recent_decode_ms=None,
            )


if __name__ == "__main__":
    unittest.main()
