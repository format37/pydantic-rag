#!/usr/bin/env python3
"""Pass 2: reconcile a Marker-generated markdown against the source PDF.

Spawns a Claude Agent SDK agent with Read + Edit + Grep tools. The agent walks
the markdown, opens the PDF visually, and patches residual conversion errors
that Marker's own LLM cleanup misses — primarily inline HTML `<sup>`/`<sub>`
fragments that should be LaTeX, detached minus signs in scientific notation,
missing subscript underscores in equations, and garbled pseudocode line
ordering inside fenced code blocks.

Uses the local `claude` CLI under your Max subscription (no API charge).

Usage:
    python scripts/cleanup_md_with_pdf.py <pdf> <md>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


SYSTEM_PROMPT = """\
You are a careful cleanup tool for scientific-paper PDF→Markdown conversions
produced by Marker. Your job is to reconcile the markdown against the source
PDF and patch *specific* classes of conversion residue. You do **not** rewrite
prose, restructure sections, or modify anything that already looks correct.

Tools you have:
- Read: opens the markdown (text) and the PDF (visual page renders).
- Grep: locate residue patterns inside the markdown quickly.
- Edit: exact-string replacement on the markdown.

Process:
1. Read the markdown end-to-end (or in chunks if it is large).
2. Read the PDF — for documents up to ~20 pages, read the whole file; for
   longer PDFs, page through the relevant ranges as needed.
3. For each residue class below, locate occurrences (Grep), verify against
   the PDF what the original typesetting actually is, then Edit only the
   confirmed bugs.
4. End with a concise summary listing each edit class and how many you applied,
   plus any cases you deliberately left alone.

Residue classes to fix:

A. **Inline HTML sub/sup fragments.** Patterns like `<sup>i</sup>`,
   `<sub>v</sub>`, or detached forms like `n 2` (where the trailing `2` is an
   orphaned superscript). When the PDF shows the index/exponent positioned as
   a subscript or superscript on a math variable, convert to inline LaTeX:
   `$w_i$`, `$x^2$`, `$\\theta_v$`, etc. **Crucially: the HTML form does not
   distinguish subscript from superscript correctly — always verify against
   the PDF whether `<sup>` is actually a superscript or whether Marker
   misclassified a subscript.**

B. **Detached minus signs in scientific notation.** Patterns like
   `10−<sup>5</sup>` or `10-5` should become `$10^{-5}$`. Confirm the sign
   is negative in the PDF.

C. **Missing subscript underscores in equations.** Patterns like
   `b_{\\theta v}` where the PDF shows `b_{\\theta_v}` (nested subscript).
   Add the underscore only when the PDF confirms it.

D. **Pseudocode line-ordering.** Inside fenced code blocks containing
   algorithms, if line numbers (`1:`, `2:`, ...) appear out of order or
   interleaved with their content, reorder so each numbered line sits with
   its content, matching the PDF.

Hard rules — do NOT touch:
- Reference list / bibliography entries.
- Tables that already look syntactically valid.
- Plain prose without a clear conversion artifact.
- Existing `$...$` or `$$...$$` blocks unless you can pinpoint a specific bug
  against the PDF.
- Image links or anchor `<span id="...">` tags.

If you are uncertain whether something is a real bug or intentional, leave it
alone and mention it in the summary instead. Conservative edits are correct
edits.
"""


USER_PROMPT_TMPL = """\
Reconcile the following markdown against the source PDF and patch residual
conversion errors per the residue classes in your system prompt.

PDF (read visually):    {pdf}
Markdown (Edit in place): {md}

Begin by reading both files. Use Grep to find candidate residue patterns
(e.g. `<sup>`, `<sub>`, `10−`, `\\theta v`). Edit only confirmed bugs. End
with a concise change-summary.
"""


async def run(pdf: Path, md: Path) -> None:
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Grep"],
        permission_mode="bypassPermissions",
        system_prompt=SYSTEM_PROMPT,
        max_turns=80,
        cwd=str(md.resolve().parent),
    )

    user_prompt = USER_PROMPT_TMPL.format(pdf=pdf.resolve(), md=md.resolve())

    edit_count = 0
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for blk in message.content:
                if isinstance(blk, TextBlock):
                    text = blk.text.strip()
                    if text:
                        print(text)
                elif isinstance(blk, ToolUseBlock):
                    if blk.name == "Edit":
                        edit_count += 1
                    print(f"[{blk.name}]", flush=True)
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None)
            tag = f"cost: ${cost:.4f}" if cost is not None else "cost: n/a"
            print(f"\n--- {edit_count} edits applied; {tag} ---")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("pdf", type=Path, help="Source PDF")
    parser.add_argument("md", type=Path, help="Marker-generated markdown to clean up")
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")
    if not args.md.is_file():
        sys.exit(f"Markdown not found: {args.md}")

    anyio.run(run, args.pdf, args.md)


if __name__ == "__main__":
    main()
