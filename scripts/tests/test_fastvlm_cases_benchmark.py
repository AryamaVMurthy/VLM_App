import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "fastvlm_cases_benchmark.py"
    spec = importlib.util.spec_from_file_location("fastvlm_cases_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ParseRunLogTest(unittest.TestCase):
    def test_parse_run_log_extracts_events_and_npu_stats(self):
        module = load_module()
        log_text = textwrap.dedent(
            """
            VLM_EVENT {"type":"TOKEN","text":"A "}
            VLM_EVENT {"type":"TOKEN","text":"forest path."}
            VLM_EVENT {"type":"BENCHMARK","ttft_sec":0.384,"turns":[{"prefill_tokens":384,"prefill_duration_ms":268.68,"prefill_tokens_per_sec":1429.2,"decode_tokens":26,"decode_duration_ms":272.47,"decode_tokens_per_sec":95.42}]}
            I0000 00:00:1772535590.438604   27108 llm_litert_npu_compiled_model_executor.cc:393] Custom NPU execution latency stats:
            Total prefill latency [us]: 268681
            (e2e) Prefill num tokens: 384
            (e2e) Prefill tokens per second: 1429.2
            (TransformerStackOnly) Prefill tokens per second: 1468.56
            Total prefill mask inference latency [us]: 2089 (0.777502%)
            Total prefill LLM inference latency [us]: 261480 (97.3199%)
            Total decode latency [us]: 272477
            Decode num tokens: 26
            Decode tokens per second: 95.4209
            (TransformerStackOnly) Decode tokens per second: 109.598
            Total decode mask inference latency [us]: 13498 (4.95381%)
            I0000 00:00:1772535590.111111   27108 session_basic.cc:177] Applied visual token budget: kept 96 of 576 projected vision tokens.
            I0000 00:00:1772535590.111222   27108 session_basic.cc:178] Visual token pruning decision: strategy=prompt_conditioned_v1 kept=96 original=576 mean_prompt_similarity=0.8125 mean_salience=0.4375 selected_token_indices=[1,4,9,12]
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(log_text)
            log_path = pathlib.Path(handle.name)

        metrics = module.parse_run_log(log_path)

        self.assertEqual(metrics["caption"], "A forest path.")
        self.assertEqual(metrics["ttft_ms"], 384.0)
        self.assertEqual(metrics["prefill_tokens_per_sec"], 1429.2)
        self.assertEqual(metrics["prefill_tokens_per_sec_transformer"], 1468.56)
        self.assertEqual(metrics["decode_tokens_per_sec"], 95.4209)
        self.assertEqual(metrics["decode_tokens_per_sec_transformer"], 109.598)
        self.assertEqual(metrics["prefill_latency_us"], 268681)
        self.assertEqual(metrics["prefill_mask_latency_us"], 2089)
        self.assertEqual(metrics["prefill_llm_latency_us"], 261480)
        self.assertEqual(metrics["decode_latency_us"], 272477)
        self.assertEqual(metrics["decode_mask_latency_us"], 13498)
        self.assertEqual(metrics["visual_tokens_kept"], 96)
        self.assertEqual(metrics["visual_tokens_original"], 576)
        self.assertEqual(metrics["visual_token_pruning_strategy"], "prompt_conditioned_v1")
        self.assertEqual(metrics["mean_prompt_similarity"], 0.8125)
        self.assertEqual(metrics["mean_salience"], 0.4375)
        self.assertEqual(metrics["selected_token_indices"], [1, 4, 9, 12])

    def test_summarize_results_groups_by_budget(self):
        module = load_module()
        rows = [
            {"budget": 0, "prefill_tokens_per_sec": 1000.0, "ttft_ms": 450.0},
            {"budget": 0, "prefill_tokens_per_sec": 1100.0, "ttft_ms": 430.0},
            {"budget": 96, "prefill_tokens_per_sec": 1300.0, "ttft_ms": 390.0},
        ]

        summary = module.summarize_results(rows, key_fields=("budget",))

        self.assertEqual(len(summary), 2)
        by_budget = {entry["budget"]: entry for entry in summary}
        self.assertEqual(by_budget[0]["count"], 2)
        self.assertEqual(by_budget[0]["prefill_tokens_per_sec_median"], 1050.0)
        self.assertEqual(by_budget[96]["ttft_ms_median"], 390.0)


if __name__ == "__main__":
    unittest.main()
