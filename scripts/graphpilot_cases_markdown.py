#!/usr/bin/env python3
"""Markdown-first paper helpers for the GraphPilot CASES builder."""

from __future__ import annotations

from pathlib import Path
import re
from typing import NamedTuple

TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_PATTERN = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


class RenderedMarkdownBundle(NamedTuple):
    section_sources: list[Path]
    section_markdown: dict[str, str]
    section_latex: dict[str, str]
    combined_markdown: str


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)
    return escaped


def _substitute_tokens(text: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in context:
            raise KeyError(
                f"Markdown paper source references missing token '{token}'. "
                "Remediation: add the token to the paper context or remove it from the markdown source."
            )
        return str(context[token])

    return TOKEN_PATTERN.sub(replace, text)


def _render_inline(text: str) -> str:
    placeholders: list[str] = []

    def hold(replacement: str) -> str:
        placeholders.append(replacement)
        return f"@@INLINE_{len(placeholders) - 1}@@"

    text = INLINE_CODE_PATTERN.sub(lambda match: hold(rf"\texttt{{{latex_escape(match.group(1))}}}"), text)
    text = BOLD_PATTERN.sub(lambda match: hold(rf"\textbf{{{latex_escape(match.group(1))}}}"), text)
    text = ITALIC_PATTERN.sub(lambda match: hold(rf"\emph{{{latex_escape(match.group(1))}}}"), text)
    escaped = latex_escape(text)
    for index, replacement in enumerate(placeholders):
        escaped = escaped.replace(latex_escape(f"@@INLINE_{index}@@"), replacement)
    return escaped


def render_markdown_section(markdown_text: str, *, abstract: bool = False) -> str:
    lines = markdown_text.splitlines()
    if any(line.strip().startswith("|") and line.strip().endswith("|") for line in lines):
        raise ValueError(
            "Pipe-table markdown is not supported in the GraphPilot CASES paper builder. "
            "Remediation: use a raw latex table block instead."
        )

    output: list[str] = []
    paragraph: list[str] = []
    list_buffer: list[str] = []
    list_kind: str | None = None
    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(_render_inline(" ".join(item.strip() for item in paragraph if item.strip())))
            output.append("")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_buffer, list_kind
        if not list_buffer:
            return
        env = "itemize" if list_kind == "ul" else "enumerate"
        output.append(rf"\begin{{{env}}}")
        for item in list_buffer:
            output.append(rf"\item {_render_inline(item)}")
        output.append(rf"\end{{{env}}}")
        output.append("")
        list_buffer = []
        list_kind = None

    def flush_code_block() -> None:
        nonlocal in_code_block, code_lang, code_lines
        if not in_code_block:
            return
        if code_lang == "latex":
            output.extend(code_lines)
            output.append("")
        else:
            output.append(r"\begin{verbatim}")
            output.extend(code_lines)
            output.append(r"\end{verbatim}")
            output.append("")
        in_code_block = False
        code_lang = ""
        code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block()
            else:
                flush_paragraph()
                flush_list()
                in_code_block = True
                code_lang = stripped[3:].strip()
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("# ") and not abstract:
            flush_paragraph()
            flush_list()
            output.append(rf"\section{{{_render_inline(stripped[2:].strip())}}}")
            output.append("")
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            output.append(rf"\subsection{{{_render_inline(stripped[3:].strip())}}}")
            output.append("")
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            output.append(rf"\subsubsection{{{_render_inline(stripped[4:].strip())}}}")
            output.append("")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if list_kind not in (None, "ul"):
                flush_list()
            list_kind = "ul"
            list_buffer.append(stripped[2:].strip())
            continue

        numbered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered_match:
            flush_paragraph()
            if list_kind not in (None, "ol"):
                flush_list()
            list_kind = "ol"
            list_buffer.append(numbered_match.group(1).strip())
            continue

        paragraph.append(line)

    flush_code_block()
    flush_paragraph()
    flush_list()
    return "\n".join(output).strip() + "\n"


def render_markdown_bundle(markdown_dir: Path, *, context: dict[str, str] | None = None) -> RenderedMarkdownBundle:
    sections_dir = markdown_dir / "sections"
    if not sections_dir.exists():
        raise FileNotFoundError(
            f"Markdown paper directory '{markdown_dir}' is missing sections/. "
            "Remediation: create markdown sections before building the CASES paper."
        )
    sources = sorted(path for path in sections_dir.glob("*.md") if path.is_file())
    if not sources:
        raise FileNotFoundError(
            f"Markdown paper directory '{sections_dir}' does not contain any .md sections. "
            "Remediation: add markdown section files before building the CASES paper."
        )

    resolved_context = context or {}
    section_markdown: dict[str, str] = {}
    section_latex: dict[str, str] = {}
    combined_parts: list[str] = []
    for source in sources:
        raw = source.read_text(encoding="utf-8")
        rendered_markdown = _substitute_tokens(raw, resolved_context)
        is_abstract = "abstract" in source.stem.lower()
        section_markdown[source.name] = rendered_markdown
        section_latex[source.name] = render_markdown_section(rendered_markdown, abstract=is_abstract)
        combined_parts.append(rendered_markdown.strip())
    combined_markdown = "\n\n".join(part for part in combined_parts if part) + "\n"
    return RenderedMarkdownBundle(
        section_sources=sources,
        section_markdown=section_markdown,
        section_latex=section_latex,
        combined_markdown=combined_markdown,
    )


__all__ = [
    "RenderedMarkdownBundle",
    "render_markdown_bundle",
    "render_markdown_section",
]
