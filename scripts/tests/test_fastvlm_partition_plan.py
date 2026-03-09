import importlib.util
import json
import pathlib
import tempfile
import textwrap
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "fastvlm_partition_plan.py"
    spec = importlib.util.spec_from_file_location("fastvlm_partition_plan", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FastVlmPartitionPlanTest(unittest.TestCase):
    def test_partition_plan_marks_precompiled_dispatch_as_subgraph_only(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            graph_path = tmpdir_path / "Section5_TFLiteModel_tf_lite_prefill_decode.full.txt"
            graph_path.write_text(
                textwrap.dedent(
                    """
                    Model Summary:
                      Num Subgraphs:   1

                    LiteRtSubgraph : [ #ops=1 #tensors=3 ] (<1x128x896xf16>) -> <1x128x896xf16>
                      LiteRtOp : [ TFL_CUSTOM_OP : DISPATCH_OP ] (<...>) -> <...>
                    """
                ).strip(),
                encoding="utf-8",
            )
            log_path = tmpdir_path / "run.log"
            log_path.write_text(
                textwrap.dedent(
                    """
                    I0000 model_resources_litert_lm.cc:67] model_type: TF_LITE_PREFILL_DECODE
                    VERBOSE: Replacing 1 out of 1 node(s) with delegate (DispatchDelegate) node, yielding 1 partitions for subgraph 0.
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = module.build_partition_plan(log_path, tmpdir_path)

            subgraph = plan["models"]["TF_LITE_PREFILL_DECODE"]["subgraphs"][0]
            self.assertEqual(subgraph["backend"], "NPU")
            self.assertFalse(subgraph["per_op_mapping_possible"])
            self.assertIn("DISPATCH_OP", subgraph["mapping_limitation"])

    def test_partition_plan_marks_partial_xnnpack_as_unresolved_per_op(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            graph_path = tmpdir_path / "Section1_TFLiteModel_tf_lite_embedder.full.txt"
            graph_path.write_text(
                textwrap.dedent(
                    """
                    Model Summary:
                      Num Subgraphs:   1

                    LiteRtSubgraph : [ #ops=4 #tensors=9 ] (<1x1xi32>) -> <1x1x896xf32>
                      LiteRtOp : [ TFL_MAXIMUM ] (<...>) -> <...>
                      LiteRtOp : [ TFL_RESHAPE ] (<...>) -> <...>
                      LiteRtOp : [ TFL_EMBEDDING_LOOKUP ] (<...>) -> <...>
                      LiteRtOp : [ TFL_RESHAPE ] (<...>) -> <...>
                    """
                ).strip(),
                encoding="utf-8",
            )
            log_path = tmpdir_path / "run.log"
            log_path.write_text(
                textwrap.dedent(
                    """
                    I0000 model_resources_litert_lm.cc:67] model_type: TF_LITE_EMBEDDER
                    VERBOSE: Replacing 1 out of 4 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 2 partitions for subgraph 0.
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = module.build_partition_plan(log_path, tmpdir_path)

            subgraph = plan["models"]["TF_LITE_EMBEDDER"]["subgraphs"][0]
            self.assertEqual(subgraph["backend"], "CPU_PARTIAL")
            self.assertFalse(subgraph["per_op_mapping_possible"])
            self.assertIn("partial", subgraph["mapping_limitation"].lower())

    def test_partition_plan_marks_full_xnnpack_subgraph_ops_as_cpu(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            graph_path = tmpdir_path / "Section3_TFLiteModel_tf_lite_vision_adapter.full.txt"
            graph_path.write_text(
                textwrap.dedent(
                    """
                    Model Summary:
                      Num Subgraphs:   1

                    LiteRtSubgraph : [ #ops=3 #tensors=8 ] (<1x256x3072xf32>) -> <1x256x896xf32>
                      LiteRtOp : [ TFL_FULLY_CONNECTED ] (<...>) -> <...>
                      LiteRtOp : [ TFL_GELU ] (<...>) -> <...>
                      LiteRtOp : [ TFL_FULLY_CONNECTED ] (<...>) -> <...>
                    """
                ).strip(),
                encoding="utf-8",
            )
            log_path = tmpdir_path / "run.log"
            log_path.write_text(
                textwrap.dedent(
                    """
                    I0000 model_resources_litert_lm.cc:67] model_type: TF_LITE_VISION_ADAPTER
                    VERBOSE: Replacing 3 out of 3 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0.
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = module.build_partition_plan(log_path, tmpdir_path)

            subgraph = plan["models"]["TF_LITE_VISION_ADAPTER"]["subgraphs"][0]
            self.assertEqual(subgraph["backend"], "CPU")
            self.assertTrue(subgraph["per_op_mapping_possible"])
            self.assertEqual([op["backend"] for op in subgraph["ops"]], ["CPU", "CPU", "CPU"])

    def test_partition_plan_handles_precompiled_then_jit_model_queue(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            (tmpdir_path / "Section4_TFLiteModel_tf_lite_vision_encoder.full.txt").write_text(
                textwrap.dedent(
                    """
                    Model Summary:
                      Num Subgraphs:   1

                    LiteRtSubgraph : [ #ops=1 #tensors=2 ] (<...>) -> <...>
                      LiteRtOp : [ TFL_CUSTOM_OP : DISPATCH_OP ] (<...>) -> <...>
                    """
                ).strip(),
                encoding="utf-8",
            )
            (tmpdir_path / "Section3_TFLiteModel_tf_lite_vision_adapter.full.txt").write_text(
                textwrap.dedent(
                    """
                    Model Summary:
                      Num Subgraphs:   1

                    LiteRtSubgraph : [ #ops=3 #tensors=8 ] (<...>) -> <...>
                      LiteRtOp : [ TFL_FULLY_CONNECTED ] (<...>) -> <...>
                      LiteRtOp : [ TFL_GELU ] (<...>) -> <...>
                      LiteRtOp : [ TFL_FULLY_CONNECTED ] (<...>) -> <...>
                    """
                ).strip(),
                encoding="utf-8",
            )
            log_path = tmpdir_path / "run.log"
            log_path.write_text(
                textwrap.dedent(
                    """
                    I0000 model_resources_litert_lm.cc:67] model_type: TF_LITE_VISION_ENCODER
                    I0000 model_resources_litert_lm.cc:67] model_type: TF_LITE_VISION_ADAPTER
                    WARNING: [compiled_model.cc:636] Compiler plugin path is provided in the environment, but the model is pre-compiled. Plugins won't be applied.
                    VERBOSE: Replacing 1 out of 1 node(s) with delegate (DispatchDelegate) node, yielding 1 partitions for subgraph 0.
                    INFO: [compiled_model.cc:908] JIT compilation changed model, reserializing...
                    VERBOSE: Replacing 3 out of 3 node(s) with delegate (DispatchDelegate) node, yielding 1 partitions for subgraph 0.
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = module.build_partition_plan(log_path, tmpdir_path)

            encoder = plan["models"]["TF_LITE_VISION_ENCODER"]["subgraphs"][0]
            self.assertEqual(encoder["backend"], "NPU")
            self.assertFalse(encoder["per_op_mapping_possible"])
            self.assertIn("DISPATCH_OP", encoder["mapping_limitation"])

            adapter = plan["models"]["TF_LITE_VISION_ADAPTER"]["subgraphs"][0]
            self.assertEqual(adapter["backend"], "NPU")
            self.assertTrue(adapter["per_op_mapping_possible"])
            self.assertEqual([op["backend"] for op in adapter["ops"]], ["NPU", "NPU", "NPU"])


if __name__ == "__main__":
    unittest.main()
