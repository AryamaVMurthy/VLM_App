import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


BENCHMARK_ROOT = pathlib.Path('/home/aryamavmurthy/work/Liquid_benchmarking_image tokens pruning')


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / 'run_fastvlm_litert_gqa_eval.py'
    spec = importlib.util.spec_from_file_location('run_fastvlm_litert_gqa_eval', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GQAEvalHelpersTest(unittest.TestCase):
    def test_load_gqa_questions_reads_expected_schema(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path = pathlib.Path(tmpdir) / 'questions.json'
            questions_path.write_text(
                json.dumps(
                    {
                        'q1': {
                            'imageId': 'image_1',
                            'question': 'Is it red?',
                            'answer': 'yes',
                        }
                    }
                ),
                encoding='utf-8',
            )

            samples = module.load_gqa_questions(questions_path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].question_id, 'q1')
        self.assertEqual(samples[0].image_id, 'image_1')
        self.assertEqual(samples[0].question, 'Is it red?')
        self.assertEqual(samples[0].answer, 'yes')

    def test_build_prediction_row_includes_exact_eval_fields(self):
        module = load_module()
        sample = module.GQASample(
            question_id='q1', image_id='image_1', question='Is it red?', answer='yes'
        )
        metrics = {
            'ttft_ms': 101.2,
            'prefill_latency_us': 250000,
            'decode_latency_us': 80000,
            'decode_tokens_per_sec': 97.5,
            'visual_tokens_kept': 96,
            'visual_tokens_original': 576,
            'visual_token_pruning_strategy': 'prompt_conditioned_v1',
            'log_path': '/tmp/run.log',
        }

        row = module.build_prediction_row(sample, 'yes', metrics)

        self.assertEqual(row['sample_id'], 'q1')
        self.assertEqual(row['prediction'], 'yes')
        self.assertEqual(row['answer'], 'yes')
        self.assertEqual(row['question'], 'Is it red?')
        self.assertEqual(row['ttft_s'], 0.1012)
        self.assertEqual(row['prefill_s'], 0.25)
        self.assertEqual(row['decode_s'], 0.08)
        self.assertEqual(row['vision_tokens'], 96)
        self.assertEqual(row['log_path'], '/tmp/run.log')

    def test_evaluate_predictions_uses_benchmark_repo_evaluator(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = pathlib.Path(tmpdir) / 'predictions.jsonl'
            rows = [
                {'prediction': 'yes', 'answer': 'yes'},
                {'prediction': 'no', 'answer': 'yes'},
            ]
            predictions_path.write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'
            )

            metrics = module.evaluate_predictions(predictions_path, BENCHMARK_ROOT)

        self.assertEqual(metrics['total_scored'], 2.0)
        self.assertEqual(metrics['correct'], 1.0)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['errors'], 0.0)

    def test_build_runner_command_respects_skip_flags(self):
        module = load_module()
        command = module.build_runner_command(
            runner_script=pathlib.Path('scripts/run_fastvlm_litert_npu_adb.sh'),
            prompt='Question: test?\nRespond exactly as: Answer: <one or two words>.',
            model_path=pathlib.Path('artifacts/models/model.litertlm'),
            max_visual_tokens=96,
            visual_token_pruning_strategy='prompt_conditioned_v1',
            max_output_tokens=12,
            constraint_regex=r' ?Answer: [A-Za-z0-9]+(?: [A-Za-z0-9]+)?',
            skip_build=False,
            skip_push=True,
        )

        self.assertIn('--skip-build', command)
        self.assertIn('0', command)
        self.assertIn('--skip-push', command)
        self.assertIn('1', command)
        self.assertIn('--prompt', command)
        self.assertIn('Question: test?\nRespond exactly as: Answer: <one or two words>.', command)
        self.assertIn('--constraint-regex', command)
        self.assertIn(r' ?Answer: [A-Za-z0-9]+(?: [A-Za-z0-9]+)?', command)

    def test_gqa_prompt_uses_structured_answer_prefix(self):
        module = load_module()
        self.assertEqual(
            module.gqa_prompt('What is it?'),
            'Question: What is it?\nRespond exactly as: Answer: <one or two words>.',
        )


if __name__ == '__main__':
    unittest.main()
