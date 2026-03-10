import importlib.util
import pathlib
import sys
import tempfile
import textwrap
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "fastvlm_host_edge_diff.py"
    spec = importlib.util.spec_from_file_location("fastvlm_host_edge_diff", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FastVlmHostEdgeDiffTest(unittest.TestCase):
    def test_parse_sample_ids_deduplicates_and_preserves_order(self):
        module = load_module()
        self.assertEqual(module._parse_sample_ids("a, b, a, c"), ["a", "b", "c"])

    def test_parse_event_stream_recovers_split_lines(self):
        module = load_module()
        log_text = textwrap.dedent(
            """
            VLM_EVENT {"type":"PREPARE_DONE","request_id":"q1","duration_ms":12.5}
            VLM_EVENT      0.0ms [  INFO ]  QnnDsp <I> QnnGraph_execute started. graph = 0x11
            {"type":"OVERLAP_RESPONSE","request_id":"q1","text":"Answer: red"}
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(log_text)
            log_path = pathlib.Path(handle.name)

        parsed = module.parse_event_stream(log_path)

        self.assertEqual(parsed.recovered_split_events, 1)
        self.assertEqual(len(parsed.events), 2)
        self.assertEqual(parsed.events[1]["type"], "OVERLAP_RESPONSE")

    def test_parse_pruning_decisions_requires_request_id(self):
        module = load_module()
        log_text = textwrap.dedent(
            """
            I0000 session_basic.cc:178] Visual token pruning decision: strategy=prompt_conditioned_v2 kept=64 original=256 mean_prompt_similarity=0.4 mean_salience=0.3 selected_token_indices=[1,2] reference_token_indices=[0,2]
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(log_text)
            log_path = pathlib.Path(handle.name)

        with self.assertRaises(RuntimeError):
            module.parse_pruning_decisions(log_path)

    def test_parse_pruning_decisions_extracts_request_scoped_metrics(self):
        module = load_module()
        log_text = textwrap.dedent(
            """
            I0000 session_basic.cc:178] Visual token pruning decision: strategy=prompt_conditioned_v2 kept=64 original=256 mean_prompt_similarity=0.4 mean_salience=0.3 mean_reference_prompt_similarity=0.35 mean_reference_salience=0.28 local_refinement_count=4 proposed_local_refinement_count=5 local_refinement_candidate_count=6 mean_local_refinement_prompt_gain=0.07 max_local_refinement_prompt_gain=0.12 controller_kept_uniform=true controller_reason=low_mean_salience selected_token_indices=[1,2] reference_token_indices=[0,2] request_id=q1
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(log_text)
            log_path = pathlib.Path(handle.name)

        decisions = module.parse_pruning_decisions(log_path)

        self.assertEqual(decisions["q1"]["local_refinement_count"], 4)
        self.assertEqual(decisions["q1"]["proposed_local_refinement_count"], 5)
        self.assertTrue(decisions["q1"]["controller_kept_uniform"])
        self.assertEqual(decisions["q1"]["controller_reason"], "low_mean_salience")
        self.assertEqual(decisions["q1"]["selected_token_indices"], [1, 2])
        self.assertEqual(decisions["q1"]["reference_token_indices"], [0, 2])


if __name__ == "__main__":
    unittest.main()
