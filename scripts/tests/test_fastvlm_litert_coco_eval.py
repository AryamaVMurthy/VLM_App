import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / 'run_fastvlm_litert_coco_eval.py'
    spec = importlib.util.spec_from_file_location('run_fastvlm_litert_coco_eval', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CocoEvalHelpersTest(unittest.TestCase):
    def test_load_coco_samples_uses_external_loader_and_cache(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src_dir = root / 'src/lfm2vl_bench/datasets'
            src_dir.mkdir(parents=True)
            (src_dir / 'coco_karpathy.py').write_text(
                """
from __future__ import annotations

def load_coco_split(split='test', max_rows=None, batch_size=100):
    rows = [
        {
            'cocoid': 7,
            'filename': 'image_7.jpg',
            'sentences': [{'raw': 'a cat on a chair'}, {'raw': 'cat sitting'}],
        }
    ]
    return rows[:max_rows]
""",
                encoding='utf-8',
            )
            image_cache = root / 'cache'
            image_cache.mkdir()
            (image_cache / 'image_7.jpg').write_bytes(b'fake-image')

            samples = module.load_coco_samples(root, 'test', 1, image_cache)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_id, '7')
        self.assertEqual(samples[0].filename, 'image_7.jpg')
        self.assertEqual(samples[0].references, ['a cat on a chair', 'cat sitting'])

    def test_build_prediction_row_includes_caption_metrics(self):
        module = load_module()
        sample = module.CocoSample(
            sample_id='7',
            image_id=7,
            filename='image_7.jpg',
            references=['a cat on a chair'],
        )
        row = module.build_prediction_row(
            sample,
            'a cat on a chair',
            {
                'ttft_ms': 200.0,
                'prefill_latency_us': 310000,
                'decode_latency_us': 91000,
                'decode_tokens_per_sec': 52.0,
                'prefill_tokens_per_sec': 101.0,
                'visual_tokens_kept': 96,
                'visual_tokens_original': 256,
                'visual_token_pruning_strategy': 'uniform',
                'log_path': '/tmp/coco.log',
            },
        )

        self.assertEqual(row['benchmark'], 'coco_karpathy')
        self.assertEqual(row['sample_id'], '7')
        self.assertEqual(row['prediction'], 'a cat on a chair')
        self.assertEqual(row['references'], ['a cat on a chair'])
        self.assertEqual(row['ttft_s'], 0.2)
        self.assertEqual(row['prefill_s'], 0.31)
        self.assertEqual(row['decode_s'], 0.091)
        self.assertEqual(row['vision_tokens'], 96)
        self.assertEqual(row['log_path'], '/tmp/coco.log')

    def test_evaluate_predictions_uses_external_coco_evaluator(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            eval_dir = root / 'src/lfm2vl_bench'
            eval_dir.mkdir(parents=True)
            (eval_dir / 'eval_coco.py').write_text(
                """
from __future__ import annotations

def evaluate_coco_jsonl(jsonl_path, work_dir):
    rows = [line for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {'CIDEr': float(len(rows)), 'ROUGE_L': 0.5}
""",
                encoding='utf-8',
            )
            predictions_path = root / 'predictions.jsonl'
            predictions_path.write_text(
                ''.join(
                    json.dumps({'sample_id': '7', 'prediction': 'caption', 'references': ['caption'], 'cocoid': 7}) + '\n'
                    for _ in range(2)
                ),
                encoding='utf-8',
            )

            metrics = module.evaluate_predictions(predictions_path, root)

        self.assertEqual(metrics['CIDEr'], 2.0)
        self.assertEqual(metrics['ROUGE_L'], 0.5)


if __name__ == '__main__':
    unittest.main()
