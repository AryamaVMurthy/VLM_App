import importlib.util
import pathlib
import sys
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / 'build_partition_aware_fastvlm_bundle.py'
    spec = importlib.util.spec_from_file_location('build_partition_aware_fastvlm_bundle', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildPartitionAwareFastVlmBundleTest(unittest.TestCase):
    def test_build_bundle_computes_quality_deltas(self):
        module = load_module()
        hardware_summary = {
            'variants': [
                {'name': 'sequential_npu_npu_uniform', 'metrics': {'device_runtime_s': 4.0}},
                {'name': 'overlap_cpu_decode_npu_prefill_uniform', 'metrics': {'event_window_s': 5.0}},
                {'name': 'overlap_cpu_decode_npu_prefill_prompt_conditioned_v2', 'metrics': {'event_window_s': 5.1, 'overlap_pipeline_speedup': 1.1}},
                {'name': 'overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2', 'metrics': {'event_window_s': 4.5, 'overlap_pipeline_speedup': 1.2}},
            ]
        }
        partition_validation = {
            'validation_passed': True,
            'config_deviation': [],
            'stage_backend_manifest': [{'stage': 'prefill', 'declared_backend': 'npu', 'observed_backend': 'NPU', 'visibility': 'opaque'}],
            'expected_transfer_edges': [{'from_stage': 'prefill', 'to_stage': 'decode', 'edge_type': 'handoff', 'cross_backend': True}],
            'unsupported_or_opaque_regions': [{'submodel': 'TF_LITE_PREFILL_DECODE', 'subgraph_index': 0, 'mapping_limitation': 'opaque'}],
            'compile_artifacts': {},
        }
        gqa_uniform = {'max_visual_tokens': 64, 'accuracy': 0.40, 'correct': 200, 'total_scored': 500, 'wall_clock_s': 1.0}
        gqa_smart = {'max_visual_tokens': 64, 'accuracy': 0.41, 'correct': 205, 'total_scored': 500, 'wall_clock_s': 1.1}
        coco_uniform = {'max_visual_tokens': 64, 'sample_count': 50, 'runtime': {'ttft_mean_s': 0.1}, 'metrics': {'CIDEr': 0.75, 'ROUGE_L': 0.45, 'Bleu_4': 0.19}}
        coco_smart = {'max_visual_tokens': 64, 'sample_count': 50, 'runtime': {'ttft_mean_s': 0.09}, 'metrics': {'CIDEr': 0.73, 'ROUGE_L': 0.44, 'Bleu_4': 0.18}}

        bundle = module.build_bundle(
            hardware_summary=hardware_summary,
            partition_validation=partition_validation,
            gqa_uniform=gqa_uniform,
            gqa_smart=gqa_smart,
            coco_uniform=coco_uniform,
            coco_smart=coco_smart,
            artifact_manifest={'hardware_summary': 'hardware.json'},
            figure_manifest={'overlap_timeline': 'timeline.svg'},
        )

        self.assertAlmostEqual(bundle['gqa']['accuracy_delta'], 0.01)
        self.assertAlmostEqual(bundle['coco']['cider_delta'], -0.02)
        self.assertTrue(bundle['partition_validation']['validation_passed'])
        self.assertEqual(len(bundle['hardware']['variants']), 4)
        report = module.build_report(bundle)
        self.assertIn('Partition-aware FastVLM bundle', report)
        self.assertIn('Accuracy delta', report)
        self.assertIn('CIDEr delta', report)
        self.assertIn('| Variant | Device window/runtime (s) |', report)
        paper = module.build_paper_draft(bundle)
        self.assertIn('## 2. System Overview', paper)
        self.assertIn('### 2.5 Partition validation tool', paper)
        self.assertIn('GQA is the primary grounded reasoning benchmark', paper)
        self.assertIn('timeline.svg', paper)


if __name__ == '__main__':
    unittest.main()
