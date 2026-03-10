import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "fastvlm_partition_validator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fastvlm_partition_validator", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FastVlmPartitionValidatorTest(unittest.TestCase):
    def test_build_validation_bundle_extracts_stage_manifest(self):
        module = load_module()
        payloads = [
            {
                "type": "OVERLAP_CONFIG",
                "prefill_backend": "npu",
                "decode_backend": "cpu",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "vision_encoder",
                "declared_backend": "npu",
                "observed_backend": "NPU_DISPATCHDELEGATE",
                "visibility": "opaque_dispatch_subgraph",
                "stage_unit": "TF_LITE_VISION_ENCODER:sg0",
                "compile_artifact": "prefill.litertlm",
                "source": "test",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "vision_adapter",
                "declared_backend": "artifact_pinned",
                "observed_backend": "CPU_XNNPACK",
                "visibility": "visible_per_op",
                "stage_unit": "TF_LITE_VISION_ADAPTER:sg0",
                "compile_artifact": "prefill.litertlm",
                "source": "test",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "pruning_seam",
                "declared_backend": "cpu_host",
                "observed_backend": "CPU_HOST",
                "visibility": "fully_visible",
                "stage_unit": "POST_PROJECTION_PRUNING_SEAM",
                "compile_artifact": "runtime",
                "source": "test",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "prefill",
                "declared_backend": "npu",
                "observed_backend": "NPU_SESSION_WITH_DISPATCH_PREFILL_SG0",
                "visibility": "opaque_dispatch_subgraph",
                "stage_unit": "TF_LITE_PREFILL_DECODE:sg0",
                "compile_artifact": "prefill.litertlm",
                "source": "test",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "decode",
                "declared_backend": "cpu",
                "observed_backend": "CPU_SESSION_RAW_FASTVLM",
                "visibility": "runtime_visible",
                "stage_unit": "RAW_FASTVLM_CPU_DECODE",
                "compile_artifact": "decode.litertlm",
                "source": "test",
            },
            {
                "type": "STAGE_BACKEND",
                "stage": "aux_mask",
                "declared_backend": "artifact_pinned",
                "observed_backend": "CPU_PARTIAL_XNNPACK_VISIBLE_AUX_MASK",
                "visibility": "partial_visible",
                "stage_unit": "TF_LITE_AUX",
                "compile_artifact": "prefill.litertlm",
                "source": "test",
            },
            {"type": "HANDOFF_EXPORT"},
            {"type": "HANDOFF_IMPORT"},
        ]
        partition_map = {
            "TF_LITE_VISION_ENCODER": {"subgraphs": [{"index": 0, "runtime_backend": "NPU_DISPATCHDELEGATE"}]},
            "TF_LITE_VISION_ADAPTER": {"subgraphs": [{"index": 0, "runtime_backend": "CPU_XNNPACK"}]},
            "TF_LITE_AUX": {
                "subgraphs": [
                    {
                        "index": 0,
                        "runtime_backend": "CPU_PARTIAL_XNNPACK",
                        "mapping_limitation": "opaque detail",
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            log_path = root / "run.log"
            log_path.write_text(
                "\n".join(f"VLM_EVENT {json.dumps(item)}" for item in payloads),
                encoding="utf-8",
            )
            partition_path = root / "partition.json"
            partition_path.write_text(json.dumps(partition_map), encoding="utf-8")
            model_path = root / "prefill.litertlm"
            model_path.write_bytes(b"prefill-model")
            decode_model_path = root / "decode.litertlm"
            decode_model_path.write_bytes(b"decode-model")

            bundle = module.build_validation_bundle(
                run_log=log_path,
                partition_json=partition_path,
                model_path=model_path,
                decode_model_path=decode_model_path,
                expected_prefill_backend="npu",
                expected_decode_backend="cpu",
            )

        self.assertTrue(bundle["validation_passed"])
        self.assertEqual(bundle["handoff_counters"]["export_event_count"], 1)
        self.assertEqual(bundle["handoff_counters"]["import_event_count"], 1)
        self.assertEqual(bundle["stage_backend_manifest"][0]["stage"], "vision_encoder")
        self.assertEqual(bundle["expected_transfer_edges"][3]["from_stage"], "prefill")
        self.assertEqual(len(bundle["unsupported_or_opaque_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
