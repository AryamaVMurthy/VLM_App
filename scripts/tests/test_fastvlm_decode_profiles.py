import importlib.util
import pathlib
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "fastvlm_decode_profiles.py"
    spec = importlib.util.spec_from_file_location("fastvlm_decode_profiles", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DecodeProfilesTest(unittest.TestCase):
    def test_short_mode_uses_brief_suffix_and_short_budget(self):
        module = load_module()

        profile = module.build_decode_profile(
            answer_mode="short",
            base_prompt="Describe this image.",
            short_max_output_tokens=48,
            long_max_output_tokens=192,
        )

        self.assertEqual(profile.answer_mode, "short")
        self.assertEqual(profile.max_output_tokens, 48)
        self.assertIn("briefly", profile.prompt.lower())

    def test_long_mode_uses_detailed_suffix_and_long_budget(self):
        module = load_module()

        profile = module.build_decode_profile(
            answer_mode="long",
            base_prompt="Describe this image.",
            short_max_output_tokens=48,
            long_max_output_tokens=192,
        )

        self.assertEqual(profile.answer_mode, "long")
        self.assertEqual(profile.max_output_tokens, 192)
        self.assertIn("detail", profile.prompt.lower())


if __name__ == "__main__":
    unittest.main()
