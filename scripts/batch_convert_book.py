#!/usr/bin/env python3
"""Pass 1 for very large PDFs: convert in page-range batches with a memory fuse.

Marker holds the *whole* document in RAM at once (it renders hi/low-res images
for every page and runs model tensors across them), so a 1000-page book can
exhaust host memory and — with little swap — hard-freeze the machine. This
orchestrator slices the PDF into page ranges and converts each range in its
own subprocess, so peak RAM is bounded by the batch size, not the book size.

Two safety features:

1. Memory fuse — a watchdog samples available system RAM while each batch runs.
   If it drops below --mem-floor-gb, the batch's whole process tree (including
   the `claude` subprocesses spawned by --use-llm) is killed *before* the
   machine thrashes, the batch size is halved, and the range is retried smaller.

2. Calibration — `--calibrate` runs a single probe batch, measures peak RAM,
   and recommends a safe batch size. You can also let the loop auto-size by
   omitting --batch-pages.

Each batch's markdown is appended to a single master `<stem>.md`; images from
all batches share the `<stem>/` folder (Marker names images by *global* page
id, so disjoint ranges never collide). Quality is preserved: every batch runs
with --use-llm by default.

Usage:
    # Calibrate first (recommended): probe 10 pages, get a batch-size suggestion
    python scripts/batch_convert_book.py datasets/RL-12/book.pdf \
        --output-dir data/documents/RL --calibrate

    # Full run, fixed batch size, prompting between batches
    python scripts/batch_convert_book.py datasets/RL-12/book.pdf \
        --output-dir data/documents/RL --batch-pages 50

    # Resume after stopping
    python scripts/batch_convert_book.py datasets/RL-12/book.pdf \
        --output-dir data/documents/RL --batch-pages 50 --start-page 301
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

logging.getLogger("pypdf").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WORKER = _PROJECT_ROOT / "scripts" / "convert_pdf.py"

GiB = 1024 ** 3


def pdf_page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def structure_scan(pdf: Path, total_ram_gb: float, mem_floor_gb: float) -> None:
    """Rough, instant heuristic: count embedded images per page (drives Marker's
    hi-res memory) and print a ballpark batch suggestion. Calibration is more
    reliable; this is just an at-a-glance hint."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    n = len(reader.pages)
    img_pages = 0
    img_total = 0
    for page in reader.pages:
        try:
            res = page.get("/Resources")
            xobj = res.get("/XObject") if res else None
            k = len(xobj) if xobj else 0
        except Exception:
            k = 0
        if k:
            img_pages += 1
            img_total += k
    density = img_total / n if n else 0
    print(
        f"  Pages: {n} | pages with images: {img_pages} "
        f"({100*img_pages/max(n,1):.0f}%) | avg XObjects/page: {density:.2f}"
    )
    # Heuristic: image-light books tolerate larger batches; image-heavy ones
    # less. Anchor at 50 pages for ~0.5 img/page, scale inversely, clamp.
    if density <= 0.2:
        hint = 80
    elif density <= 0.6:
        hint = 50
    elif density <= 1.2:
        hint = 30
    else:
        hint = 20
    print(
        f"  Structure-based hint: ~{hint} pages/batch "
        f"(rough — confirm with --calibrate)."
    )


def run_batch(
    pdf: Path,
    output_dir: Path,
    lo: int,
    hi: int,
    mem_floor_gb: float,
    use_llm: bool,
    poll_s: float = 0.5,
) -> tuple[str, float, Path | None]:
    """Convert pages [lo, hi] (1-indexed inclusive) in a killable subprocess.

    A watchdog samples available RAM; if it falls below mem_floor_gb the whole
    process group is killed. Returns (status, peak_used_gb, partial_md_path):
      "ok"    — converted; partial_md_path is the written markdown.
      "fuse"  — a memory event: our watchdog tripped, OR the OS OOM killer
                SIGKILLed the worker (a sub-poll-interval spike the watchdog
                can miss). Retrying with a smaller batch is the right response.
      "crash" — a deterministic worker failure (exception / nonzero exit);
                retrying smaller won't help.
    """
    stem = pdf.stem
    part_name = f"{stem}.part_{lo:05d}-{hi:05d}.md"
    part_path = output_dir / part_name

    cmd = [
        sys.executable, "-u", str(_WORKER), str(pdf),
        "--output-dir", str(output_dir),
        "--page-range", f"{lo}-{hi}",
        "--md-name", part_name,
    ]
    if use_llm:
        cmd.append("--use-llm")

    total_ram = psutil.virtual_memory().total
    min_available = total_ram  # track the worst dip
    fuse_tripped = False

    # start_new_session so the worker + its `claude` children share a process
    # group we can kill atomically.
    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = os.getpgid(proc.pid)

    try:
        while proc.poll() is None:
            avail = psutil.virtual_memory().available
            if avail < min_available:
                min_available = avail
            if avail < mem_floor_gb * GiB:
                fuse_tripped = True
                print(
                    f"\n  [FUSE] Available RAM {avail/GiB:.1f} GiB < floor "
                    f"{mem_floor_gb:.1f} GiB — killing batch {lo}-{hi} before "
                    f"the machine thrashes.",
                    file=sys.stderr,
                )
                _kill_group(pgid)
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        _kill_group(pgid)
        raise
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _kill_group(pgid, hard=True)

    peak_used_gb = (total_ram - min_available) / GiB

    # An OS OOM kill (SIGKILL) we didn't initiate is also a memory event: the
    # watchdog samples every poll_s and can miss a sub-interval spike that the
    # kernel OOM killer catches first. Trust the signal over the sampled peak.
    if not fuse_tripped and proc.returncode == -signal.SIGKILL:
        fuse_tripped = True

    if fuse_tripped:
        part_path.unlink(missing_ok=True)
        return "fuse", peak_used_gb, None
    if proc.returncode == 0 and part_path.is_file():
        return "ok", peak_used_gb, part_path
    # Deterministic worker failure (exception / nonzero exit, not a signal).
    part_path.unlink(missing_ok=True)
    return "crash", peak_used_gb, None


def _kill_group(pgid: int, hard: bool = False) -> None:
    sig = signal.SIGKILL if hard else signal.SIGTERM
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    if not hard:
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def append_master(master: Path, part: Path) -> None:
    text = part.read_text(encoding="utf-8")
    if master.exists() and master.stat().st_size:
        with master.open("a", encoding="utf-8") as fh:
            fh.write("\n\n")
            fh.write(text)
    else:
        master.write_text(text, encoding="utf-8")


def recommend_batch(peak_used_gb: float, pages: int, total_ram_gb: float,
                    mem_floor_gb: float) -> int:
    """Conservative linear estimate: assume peak scales from zero with pages
    (counts fixed model overhead as if per-page, so it under-shoots = safe)."""
    budget = total_ram_gb - mem_floor_gb
    per_page = peak_used_gb / max(pages, 1)
    if per_page <= 0:
        return 50
    return max(5, int(budget / per_page))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pass 1 for large PDFs: page-range batched Marker conversion "
        "with a memory fuse and calibration."
    )
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("data/documents"))
    ap.add_argument(
        "--batch-pages", type=int, default=None,
        help="Pages per batch. Omit to auto-size from a calibration probe.",
    )
    ap.add_argument(
        "--start-page", type=int, default=1,
        help="First page (1-indexed) to convert. Use to resume; appends to the "
        "existing master .md.",
    )
    ap.add_argument(
        "--end-page", type=int, default=None,
        help="Last page (1-indexed, inclusive) to convert. Defaults to the last "
        "page. Pair with --start-page to cap a run for quota pacing.",
    )
    ap.add_argument(
        "--mem-floor-gb", type=float, default=5.0,
        help="Kill a batch if available RAM drops below this (default 5).",
    )
    ap.add_argument(
        "--min-batch", type=int, default=5,
        help="Smallest batch the fuse will shrink to before giving up.",
    )
    ap.add_argument(
        "--calibrate", nargs="?", type=int, const=10, default=None,
        metavar="N",
        help="Probe N pages (default 10), measure peak RAM, recommend a batch "
        "size, then exit without converting the whole book.",
    )
    ap.add_argument("--scan", action="store_true",
                    help="Print an instant structure-based batch hint and exit.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Disable Marker's --use-llm (NOT recommended; lowers "
                    "table/equation quality). Default is use-llm ON.")
    ap.add_argument("--yes", action="store_true",
                    help="Don't prompt between batches; run straight through.")
    args = ap.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    use_llm = not args.no_llm
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / GiB
    total = pdf_page_count(args.pdf)
    end_page = min(args.end_page or total, total)

    print(
        f"Book: {args.pdf.name}\n"
        f"  Total pages: {total} | converting {args.start_page}-{end_page}\n"
        f"  Host RAM: {total_ram_gb:.1f} GiB | mem floor: {args.mem_floor_gb} GiB | "
        f"swap: {psutil.swap_memory().total/GiB:.1f} GiB\n"
        f"  use-llm: {use_llm}"
    )

    if args.scan:
        structure_scan(args.pdf, total_ram_gb, args.mem_floor_gb)
        return

    if args.calibrate is not None:
        n = args.calibrate
        lo, hi = args.start_page, min(args.start_page + n - 1, end_page)
        print(f"\n=== Calibration probe: pages {lo}-{hi} ({hi-lo+1} pages) ===")
        t0 = time.time()
        status, peak, part = run_batch(
            args.pdf, args.output_dir, lo, hi, args.mem_floor_gb, use_llm
        )
        dt = time.time() - t0
        if status != "ok":
            print(
                f"\nProbe {status.upper()} at {hi-lo+1} pages, peak ~{peak:.1f} GiB. "
                f"Try a smaller probe: --calibrate 5",
                file=sys.stderr,
            )
            sys.exit(1)
        # Keep the probe's output as the start of the master so it isn't wasted.
        master = args.output_dir / f"{args.pdf.stem}.md"
        append_master(master, part)
        part.unlink(missing_ok=True)
        rec = recommend_batch(peak, hi - lo + 1, total_ram_gb, args.mem_floor_gb)
        print(
            f"\nProbe OK in {dt:.0f}s. Peak RAM used: ~{peak:.1f} GiB for "
            f"{hi-lo+1} pages (~{peak/(hi-lo+1):.2f} GiB/page incl. model load).\n"
            f"Recommended batch size: ~{rec} pages "
            f"(keeps peak under {total_ram_gb - args.mem_floor_gb:.0f} GiB).\n"
            f"Probe output appended to {master.name}. To convert the rest:\n"
            f"  python scripts/batch_convert_book.py {args.pdf} "
            f"--output-dir {args.output_dir} --batch-pages {rec} "
            f"--start-page {hi+1}"
        )
        return

    batch = args.batch_pages
    if batch is None:
        sys.exit(
            "No --batch-pages given. Run --calibrate first to get a suggestion, "
            "or pass --batch-pages explicitly."
        )
    batch = max(args.min_batch, batch)

    master = args.output_dir / f"{args.pdf.stem}.md"
    if args.start_page == 1 and master.exists():
        print(f"\n[warn] {master} exists and start-page is 1 — it will be "
              f"overwritten by this run.")
        master.unlink()

    start = args.start_page
    converted_to = start - 1
    while start <= end_page:
        hi = min(start + batch - 1, end_page)
        print(f"\n=== Batch: pages {start}-{hi} of {end_page} "
              f"(size {batch}) ===")
        t0 = time.time()
        status, peak, part = run_batch(
            args.pdf, args.output_dir, start, hi, args.mem_floor_gb, use_llm
        )
        dt = time.time() - t0

        if status == "fuse" and batch > args.min_batch:
            # Memory event — shrink and retry the SAME start (no page skipped).
            new_batch = max(args.min_batch, batch // 2)
            print(
                f"  [FUSE] Batch {start}-{hi} hit the memory floor "
                f"(peak ~{peak:.1f} GiB). Shrinking {batch} -> {new_batch} "
                f"and retrying.",
                file=sys.stderr,
            )
            batch = new_batch
            continue
        if status == "fuse":
            print(
                f"\n[error] Memory fuse tripped even at the minimum batch size "
                f"({args.min_batch} pages), peak ~{peak:.1f} GiB.\n"
                f"  Add swap or lower --mem-floor-gb, then resume with:\n"
                f"  --start-page {start} --batch-pages {args.min_batch}",
                file=sys.stderr,
            )
            sys.exit(1)
        if status == "crash":
            print(
                f"\n[error] Worker crashed converting pages {start}-{hi} "
                f"(not a memory event — retrying smaller won't help).\n"
                f"  Inspect the worker output above for the cause. "
                f"Resume after fixing with:\n"
                f"  --start-page {start} --batch-pages {batch}",
                file=sys.stderr,
            )
            sys.exit(1)

        append_master(master, part)
        part.unlink(missing_ok=True)
        converted_to = hi
        print(f"  OK in {dt:.0f}s — peak ~{peak:.1f} GiB. "
              f"Appended to {master.name}.")

        if hi >= end_page:
            print(f"\nAll pages {args.start_page}-{end_page} converted into "
                  f"{master}.")
            break

        next_start = hi + 1
        remaining = end_page - hi
        if args.yes:
            start = next_start
            continue
        try:
            ans = input(
                f"\nProgress: pages {args.start_page}-{hi}/{end_page} done.\n"
                f"Next batch starts at page {next_start} ({remaining} remaining).\n"
                f"  [Enter]   continue with batch size {batch}\n"
                f"  [number]  continue with a new batch size\n"
                f"  [n / Ctrl-D]  stop\n"
                f"> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans.lower() in {"n", "no", "stop", "q", "quit"}:
            print(f"\nStopped at page {hi}. Resume with:\n"
                  f"  --start-page {next_start} --batch-pages {batch}")
            break
        if ans:
            try:
                batch = max(args.min_batch, int(ans))
            except ValueError:
                print(f"  (couldn't parse '{ans}', keeping batch size {batch})")
        start = next_start


if __name__ == "__main__":
    main()
