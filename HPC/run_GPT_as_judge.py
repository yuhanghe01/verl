"""Driver: run summary_judge.judge_summary over eval_predictions JSONL files.

Field mapping for these files:
  user       -> prompt (incident thread)
  reference  -> reference_summary
  prediction -> candidate_summary

Only rows with task_type == "summarization" are judged. Other rows pass through
unchanged. Judged results are written to <name>.judged.jsonl and a per-file
mean-score summary is printed.

Usage:
  python run_judge.py --limit 3 FILE.jsonl          # smoke test
  python run_judge.py FILE1.jsonl FILE2.jsonl ...    # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from summary_judge import SummaryJudgeConfig, judge_summary

TARGET_TASK = "summarization"


def score_row(idx: int, row: dict, config: SummaryJudgeConfig) -> tuple[int, dict | None, str | None]:
    """Return (idx, judge_result_or_None, error_or_None)."""
    try:
        result = judge_summary(
            prompt=row.get("user", "") or "",
            reference_summary=row.get("reference", "") or "",
            candidate_summary=row.get("prediction", "") or "",
            config=config,
        )
        return idx, result, None
    except Exception as exc:  # noqa: BLE001 - keep the batch alive
        return idx, None, f"{type(exc).__name__}: {exc}"


def process_file(path: Path, config: SummaryJudgeConfig, limit: int | None, workers: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summ_idx = [i for i, r in enumerate(rows) if r.get("task_type") == TARGET_TASK]
    if limit is not None:
        summ_idx = summ_idx[:limit]

    print(f"\n=== {path.name} : {len(summ_idx)} summarization rows to judge "
          f"(of {len(rows)} total) ===", flush=True)

    done = 0
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(score_row, i, rows[i], config): i for i in summ_idx}
        for fut in as_completed(futures):
            idx, result, err = fut.result()
            if err is not None:
                errors += 1
                rows[idx]["gpt52_judge_error"] = err
                print(f"  [row {idx}] ERROR {err[:160]}", flush=True)
            else:
                rows[idx]["gpt52_judge"] = result
            done += 1
            if done % 10 == 0 or done == len(summ_idx):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  progress {done}/{len(summ_idx)}  ({rate:.2f} rows/s, {errors} errors)", flush=True)

    # Write judged output (only when not a dry/limited preview write? always write).
    out_path = path.with_suffix(".judged.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate stats over successfully judged rows.
    scored = [rows[i]["gpt52_judge"] for i in summ_idx if "gpt52_judge" in rows[i]]
    n = len(scored)

    def mean(key_path):
        vals = []
        for s in scored:
            v = s
            for k in key_path:
                v = v.get(k, {}) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "file": path.name,
        "judged": n,
        "errors": errors,
        "primary_score": mean(["primary_score"]),
        "factual_correctness": mean(["summary_judge", "factual_correctness"]),
        "coverage": mean(["summary_judge", "coverage"]),
        "no_hallucination": mean(["summary_judge", "no_hallucination"]),
        "concision": mean(["summary_judge", "concision"]),
        "out_path": str(out_path),
    }
    print(f"  -> wrote {out_path.name}", flush=True)
    print(f"  -> mean primary_score = {summary['primary_score']:.4f} "
          f"(factual {summary['factual_correctness']:.3f}, coverage {summary['coverage']:.3f}, "
          f"no_halluc {summary['no_hallucination']:.3f}, concision {summary['concision']:.3f})", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--mode", default="trapi", choices=["trapi", "none"])
    ap.add_argument("--model", default="gpt-5.2_2025-12-11")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="judge only first N summarization rows per file")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    config = SummaryJudgeConfig(mode=args.mode, model=args.model, trials=args.trials)
    print(f"config: mode={config.mode} model={config.model} trials={config.trials} "
          f"workers={args.workers} limit={args.limit}", flush=True)

    summaries = []
    for path in args.files:
        if not path.exists():
            print(f"!! missing: {path}", file=sys.stderr)
            continue
        summaries.append(process_file(path, config, args.limit, args.workers))

    print("\n================ OVERALL SUMMARY ================", flush=True)
    print(f"{'model file':52} {'judged':>6} {'err':>4} {'primary':>8}", flush=True)
    for s in summaries:
        print(f"{s['file']:52} {s['judged']:>6} {s['errors']:>4} {s['primary_score']:>8.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 