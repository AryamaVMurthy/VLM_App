import importlib.util
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "graphpilot_cases_markdown.py"
    assert module_path.exists(), f"Missing markdown helper: {module_path}"
    spec = importlib.util.spec_from_file_location("graphpilot_cases_markdown", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GraphPilotCasesMarkdownTest(unittest.TestCase):
    def test_markdown_renderer_supports_headings_lists_and_raw_latex(self) -> None:
        module = load_module()
        markdown_text = (
            "# Introduction\n\n"
            "GraphPilot keeps **support-safe** execution and `kv_cache` accounting explicit.\n\n"
            "- runtime scheduler\n"
            "- calibrated simulator\n\n"
            "```latex\n"
            "\\begin{figure}[t]\n"
            "\\caption{Raw latex figure.}\n"
            "\\end{figure}\n"
            "```\n"
        )

        latex = module.render_markdown_section(markdown_text)

        self.assertIn("\\section{Introduction}", latex)
        self.assertIn("\\textbf{support-safe}", latex)
        self.assertIn("\\texttt{kv\\_cache}", latex)
        self.assertIn("\\begin{itemize}", latex)
        self.assertIn("\\caption{Raw latex figure.}", latex)

    def test_markdown_renderer_preserves_raw_latex_citations(self) -> None:
        module = load_module()
        markdown_text = (
            "# Related Work\n\n"
            "```latex\n"
            "Band~\\cite{band2022} and Twill~\\cite{twill2025} are cited in raw latex.\n"
            "```\n"
        )

        latex = module.render_markdown_section(markdown_text)

        self.assertIn("Band~\\cite{band2022}", latex)
        self.assertIn("Twill~\\cite{twill2025}", latex)

    def test_markdown_renderer_fails_fast_on_pipe_tables(self) -> None:
        module = load_module()

        with self.assertRaises(ValueError):
            module.render_markdown_section("| bad |\n| --- |\n| table |\n")

    def test_markdown_bundle_renders_sections_and_combined_markdown(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sections_dir = root / "sections"
            sections_dir.mkdir()
            (sections_dir / "01_intro.md").write_text("# Intro\n\nParagraph.\n", encoding="utf-8")
            (sections_dir / "02_results.md").write_text("# Results\n\n- one\n- two\n", encoding="utf-8")

            rendered = module.render_markdown_bundle(root)

            self.assertEqual([item.stem for item in rendered.section_sources], ["01_intro", "02_results"])
            self.assertIn("# Intro", rendered.combined_markdown)
            self.assertIn("\\section{Results}", rendered.section_latex["02_results.md"])


if __name__ == "__main__":
    unittest.main()
