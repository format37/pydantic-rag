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
import logging
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

# pypdf prints "Ignoring wrong pointing object" warnings for minor PDF
# structural quirks. Harmless, but noisy for large books.
logging.getLogger("pypdf").setLevel(logging.ERROR)


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


USER_PROMPT_BATCH_TMPL = """\
Reconcile a *page range* of the markdown against the source PDF. Patch only
residual conversion errors per the residue classes in your system prompt, and
**only within the markdown section that corresponds to the given page range**.

PDF:               {pdf}
Markdown:          {md}
Page range:        {start}–{end} (1-indexed) of {total} total PDF pages

Process:
1. Read the markdown. Use Grep to find `<span id="page-N-` anchor markers.
   Verify the anchor's indexing convention by spot-checking one anchor against
   the PDF (Marker's anchors may be 0- or 1-indexed depending on the document).
2. From the anchors, identify the markdown line range that corresponds to PDF
   pages {start}–{end}.
3. Read the relevant PDF pages (`Read` accepts at most 20 pages per call —
   make multiple calls for larger ranges).
4. Edit ONLY within that markdown line range. Do not touch content outside it.
5. End with a concise summary listing the line-range you scoped to, fixes
   applied per residue class, and anything you deliberately skipped.
"""


async def run(
    pdf: Path,
    md: Path,
    start_page: int | None = None,
    end_page: int | None = None,
    total_pages: int | None = None,
) -> dict:
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Grep"],
        permission_mode="bypassPermissions",
        system_prompt=SYSTEM_PROMPT,
        max_turns=80,
        cwd=str(md.resolve().parent),
        # Reading multi-page PDFs returns image tokens whose JSON-encoded
        # tool result can exceed the SDK's default 1 MB buffer. Bump high.
        max_buffer_size=64 * 1024 * 1024,
    )

    if start_page is not None and end_page is not None:
        user_prompt = USER_PROMPT_BATCH_TMPL.format(
            pdf=pdf.resolve(),
            md=md.resolve(),
            start=start_page,
            end=end_page,
            total=total_pages or "?",
        )
    else:
        user_prompt = USER_PROMPT_TMPL.format(pdf=pdf.resolve(), md=md.resolve())

    edit_count = 0
    batch_cost = 0.0
    saw_result = False
    try:
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
                saw_result = True
                cost = getattr(message, "total_cost_usd", None)
                if cost is not None:
                    batch_cost = float(cost)
                tag = f"cost: ${cost:.4f}" if cost is not None else "cost: n/a"
                print(f"\n--- {edit_count} edits applied; {tag} ---")
    except Exception as exc:
        # The claude CLI sometimes emits a non-zero exit error AFTER delivering
        # the ResultMessage. If we already received the result, the edits are
        # in and we should not crash the outer batch loop.
        if saw_result:
            print(f"  [note] SDK post-completion error swallowed: {exc}", file=sys.stderr)
        else:
            raise

    return {"edits": edit_count, "cost": batch_cost}


def _pdf_page_count(pdf: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(pdf).pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("pdf", type=Path, help="Source PDF")
    parser.add_argument("md", type=Path, help="Marker-generated markdown to clean up")
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=None,
        help="Cleanup PDF pages in batches of N. Without this flag, the agent "
        "walks the whole document at once (best for ≤100-page docs). With it, "
        "you'll be prompted between batches and can adjust the size on the fly.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="First page (1-indexed) to process when --batch-pages is set.",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")
    if not args.md.is_file():
        sys.exit(f"Markdown not found: {args.md}")

    if args.batch_pages is None:
        anyio.run(run, args.pdf, args.md)
        return

    total = _pdf_page_count(args.pdf)
    start = max(1, args.start_page)
    batch = max(1, args.batch_pages)

    total_edits = 0
    total_cost = 0.0

    while start <= total:
        end = min(start + batch - 1, total)
        print(f"\n=== Cleanup batch: pages {start}-{end} of {total} ===")
        try:
            stats = anyio.run(run, args.pdf, args.md, start, end, total)
            total_edits += stats.get("edits", 0)
            total_cost += stats.get("cost", 0.0)
        except Exception as exc:
            print(
                f"\n[error] Batch pages {start}-{end} failed before completion: {exc}\n"
                f"  Resume with: --start-page {start} --batch-pages {batch}",
                file=sys.stderr,
            )
            break

        if end >= total:
            print(
                f"\nAll {total} pages processed. "
                f"Cumulative: {total_edits} edits, ${total_cost:.4f}."
            )
            break

        next_start = end + 1
        remaining = total - end
        try:
            ans = input(
                f"\nProgress: pages 1-{end}/{total} done — "
                f"{total_edits} edits total, ${total_cost:.4f} spent.\n"
                f"Next batch starts at page {next_start} ({remaining} remaining).\n"
                f"  [Enter]   continue with batch size {batch}\n"
                f"  [number]  continue with a new batch size\n"
                f"  [n / Ctrl-D]  stop\n"
                f"> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans.lower() in {"n", "no", "stop", "q", "quit"}:
            print(
                f"\nStopped at page {end}. Resume with:\n"
                f"  --start-page {next_start} --batch-pages {batch}"
            )
            break

        if ans:
            try:
                batch = max(1, int(ans))
            except ValueError:
                print(f"  (couldn't parse '{ans}', keeping batch size {batch})")

        start = next_start


if __name__ == "__main__":
    main()
