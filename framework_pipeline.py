"""
framework_pipeline.py — Runs the full hero type classification pipeline in order.

You can control which steps and models to run using command-line arguments. 
By default, it runs all steps for both Gemini and Qwen, and compares against both Seal & White's index and manual annotations.

Usage:
    python framework_pipeline.py                         # all steps, both models
    python framework_pipeline.py --models gemini         # only Gemini
    python framework_pipeline.py --models qwen           # only Qwen
    python framework_pipeline.py --skip-eval             # skip evaluation, post-process only
    python framework_pipeline.py --steps 3 4 5 6         # specific steps only, you can combine steps as needed 
                                                           (e.g. --steps 1 3 to run Gemini evaluation and then build the graph, skipping Qwen and comparisons or just --steps 5 to only run comparisons against Seal&White's ground truth)
    python framework_pipeline.py --ground-truth seal     # compare against Seal & White only
    python framework_pipeline.py --ground-truth manual   # compare against manual annotations only
    python framework_pipeline.py --ground-truth both     # compare against both (default)

Note: you can combine arguments to run specific models on specific steps (e.g.: python framework_pipeline.py --models gemini --steps 1 3 5 --ground-truth seal).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def banner(title: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*70}\n", flush=True)

def run(cmd: list) -> None:
    """Run a pipeline script as a subprocess."""
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        log(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hero Type Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  evaluate_framework_gemini_batch.py   Gemini batch evaluation
  2  evaluate_framework_two-step.py       Qwen3 vLLM evaluation
  3  build_result_table.py                Build results CSV
  4  build_knowledge_graph.py             Build knowledge graph
  5  compare_against_ground_truth.py      Compare against Seal & White index
  6  compare_against_ground_truth.py      Compare against manual annotations
        """
    )
    parser.add_argument(
        "--models", nargs="+", choices=["gemini", "qwen", "both"], default=["both"],
        help="Which model(s) to run (default: both)"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Which steps to run (default: all)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation steps 1 & 2, only run post-processing"
    )
    parser.add_argument(
        "--ground-truth", choices=["seal", "manual", "both"], default="both",
        help="Which ground truth to compare against in steps 5 & 6 (default: both)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve steps
    if args.steps:
        steps = set(args.steps)
    elif args.skip_eval:
        steps = {3, 4, 5, 6}
    else:
        steps = {1, 2, 3, 4, 5, 6}

    # Resolve models
    models = set()
    for m in args.models:
        if m == "both":
            models.update(["gemini", "qwen"])
        else:
            models.add(m)

    log(f"Models:       {sorted(models)}")
    log(f"Steps:        {sorted(steps)}")
    log(f"Ground truth: {args.ground_truth}")

    pipeline_dir = Path("pipeline")

    # ── Step 1: Gemini evaluation ─────────────────────────────────────────────
    if 1 in steps and "gemini" in models:
        banner("STEP 1 — Gemini Batch Evaluation")
        run([pipeline_dir / "evaluate_framework_gemini_batch.py"])

    # ── Step 2: Qwen evaluation ───────────────────────────────────────────────
    if 2 in steps and "qwen" in models:
        banner("STEP 2 — Qwen3 Two-Step Evaluation")
        run([pipeline_dir / "evaluate_framework_two-step.py"])

    # ── Step 3: Build results table ───────────────────────────────────────────
    if 3 in steps:
        banner("STEP 3 — Build Results Table")
        for model in sorted(models):
            run([pipeline_dir / "build_result_table.py", "--model", model])

    # ── Step 4: Build knowledge graph ─────────────────────────────────────────
    if 4 in steps:
        banner("STEP 4 — Build Knowledge Graph")
        for model in sorted(models):
            run([pipeline_dir / "build_knowledge_graph.py", "--model", model])

    # ── Step 5: Compare against Seal & White index ────────────────────────────
    if 5 in steps and args.ground_truth in ("seal", "both"):
        banner("STEP 5 — Compare Against Seal & White Index")
        for model in sorted(models):
            run([
                pipeline_dir / "compare_against_ground_truth.py",
                "--model", model,
                "--ground-truth", "seal-white",
            ])

    # ── Step 6: Compare against manual annotations ────────────────────────────
    if 6 in steps and args.ground_truth in ("manual", "both"):
        banner("STEP 6 — Compare Against Manual Annotations")
        for model in sorted(models):
            run([
                pipeline_dir / "compare_against_ground_truth.py",
                "--model", model,
                "--ground-truth", "manual",
            ])

    banner("Pipeline complete.")


if __name__ == "__main__":
    main()