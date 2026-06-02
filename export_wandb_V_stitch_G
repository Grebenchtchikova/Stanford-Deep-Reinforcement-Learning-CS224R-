"""
W&B -> results.json export with exact stitching for crashed/resumed runs.

Usage:
    python export_wandb.py

Key behavior:
    - Normal experiments: export longest matching run.
    - math-clean: explicitly stitch:
        uojla9ie:  steps 0-299
        hkvxuwgl:  steps 300-400

This avoids:
    - missing crashed runs because experiment metadata differs
    - double-counting overlapping steps 300-399
    - accidentally letting the crashed run override the resumed run
"""

import json
import wandb


ENTITY = "greben-stanford-university"

TRAINING_PROJECTS = {
    "cs224r-trivia-gsm8k": [
        "gsm8k-clean",
        "gsm8k-trivia-only",
        "gsm8k-partial-e3-clean",
        "gsm8k-partial-e3-trivia",
        "gsm8k-track-e-2M-partial-e3",
    ],
    "cs224r-trivia-hendrycks": [
        "math-clean",
        "math-trivia-only",
    ],
}

AIME_PROJECT = "cs224r-trivia-aime"


# Exact/manual stitching config.
# This is safer than discovering by experiment_name because crashed/resumed runs
# can have weird names/configs.
STITCHED_EXPERIMENTS = {
    "math-clean": [
        {
            "run_id": "uojla9ie",
            "start": 0,
            "end": 299,
            "step_offset": 0,
            "label": "original_0_299",
        },
        {
            "run_id": "hkvxuwgl",
            "start": 300,
            "end": 400,
            "step_offset": 0,
            "label": "resume_300_400",
        },
    ],
}


# Optional generic merge experiments.
# Keep empty for now because math-clean should use exact stitching, not generic merge.
MERGE_EXPERIMENTS = set()


KEEP = [
    "actor/entropy",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "response_length/clip_ratio",
    "response_length/mean",
    "response_length/min",
    "response_length/max",
    "timing_s/testing",
    "timing_s/training",
    "timing_s/step",
    "timing_s/gen",
    "critic/score/mean",
    "critic/rewards/mean",
    "test_score",
    "reward/mean",
    "length/mean",
    "length/unknown",
    "perf/max_memory_allocated_gb",
    "steps/compute_matched_steps",
    "_step",
]


def should_keep(key):
    """Return True if a W&B history key should be exported."""
    return any(k in key for k in KEEP)


def json_safe(value):
    """Convert non-JSON-serializable values to strings."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def get_all_runs(api, project, exp_name):
    """
    Get all runs matching an experiment name.

    First tries config.trainer.experiment_name.
    Falls back to run.name.
    """
    runs = list(
        api.runs(
            f"{ENTITY}/{project}",
            filters={"config.trainer.experiment_name": exp_name},
        )
    )

    if not runs:
        all_runs = list(api.runs(f"{ENTITY}/{project}"))
        runs = [r for r in all_runs if r.name == exp_name]

    return runs


def get_longest_run(api, project, exp_name):
    """Return the matching run with the highest summary _step."""
    runs = get_all_runs(api, project, exp_name)

    if not runs:
        print(f"  WARNING: no runs for {exp_name}")
        return None

    best = max(runs, key=lambda r: r.summary.get("_step", 0) or 0)

    print(
        f"  {exp_name}: selected longest run "
        f"name={best.name} id={best.id} steps={best.summary.get('_step', 0)}"
    )

    return best


def get_run_by_id(api, project, run_id):
    """Fetch a W&B run by exact run ID."""
    return api.run(f"{ENTITY}/{project}/{run_id}")


def export_run(run):
    """
    Pull all history from a single run and filter client-side.

    Multiple W&B history rows may exist per _step; this function preserves rows
    as-is, matching your original behavior.
    """
    rows_out = []

    for row in run.scan_history(page_size=500):
        filtered = {}

        for k, v in row.items():
            if v is not None and should_keep(k):
                filtered[k] = v

        if filtered:
            filtered["_step"] = row.get("_step", len(rows_out))
            rows_out.append(filtered)

    print(f"    {len(rows_out)} rows, {sum(len(r) for r in rows_out)} values")

    return rows_out


def export_run_segment(run, start, end, step_offset=0, label=None):
    """
    Pull a precise step window from one run.

    This also merges multiple partial W&B rows for the same logical _step into
    one row. Later rows for the same step update/fill the same row.

    Args:
        run: W&B run object.
        start: first logical step to keep.
        end: last logical step to keep.
        step_offset:
            Add this to raw W&B _step before filtering.
            Use 0 if the resumed run already logs 300-400.
            Use 300 if the resumed run restarted at raw steps 0-100.
        label: provenance label.
    """
    rows_by_step = {}
    raw_rows = 0
    kept_history_rows = 0

    for row in run.scan_history(page_size=500):
        raw_rows += 1

        raw_step = row.get("_step")
        if raw_step is None:
            continue

        try:
            raw_step = int(raw_step)
        except (TypeError, ValueError):
            continue

        logical_step = raw_step + step_offset

        if logical_step < start or logical_step > end:
            continue

        filtered = {}

        for k, v in row.items():
            if v is not None and should_keep(k):
                filtered[k] = v

        if not filtered:
            continue

        # Force canonical stitched step.
        filtered["_step"] = logical_step

        # Provenance fields. These are intentionally kept even though they are
        # not in KEEP, because they make the stitch auditable.
        filtered["_source_run_id"] = run.id
        filtered["_source_run_name"] = run.name
        filtered["_source_raw_step"] = raw_step

        if label:
            filtered["_source_segment"] = label

        if logical_step not in rows_by_step:
            rows_by_step[logical_step] = {}

        # Merge partial rows for the same step.
        rows_by_step[logical_step].update(filtered)
        kept_history_rows += 1

    rows = [rows_by_step[s] for s in sorted(rows_by_step)]

    print(
        f"    segment={label or run.id}: "
        f"raw_rows={raw_rows}, "
        f"kept_history_rows={kept_history_rows}, "
        f"unique_steps={len(rows)}, "
        f"step_range={start}-{end}"
    )

    return rows


def find_test_score(row):
    """Return the first visible test_score metric in a row, if present."""
    for k, v in row.items():
        if "test_score" in k and isinstance(v, (int, float)):
            return k, v

    return None, None


def validate_stitched_rows(rows, exp_name):
    """Print sanity checks for stitched rows."""
    if not rows:
        print(f"    WARNING: {exp_name} produced zero stitched rows")
        return

    steps = [int(r["_step"]) for r in rows]
    min_step = min(steps)
    max_step = max(steps)

    expected = set(range(min_step, max_step + 1))
    actual = set(steps)
    missing = sorted(expected - actual)
    duplicates = len(steps) - len(actual)

    print(
        f"    stitched result: {len(rows)} rows, "
        f"range={min_step}-{max_step}, "
        f"duplicates={duplicates}"
    )

    if missing:
        preview = missing[:20]
        suffix = "..." if len(missing) > 20 else ""
        print(f"    WARNING: missing {len(missing)} steps: {preview}{suffix}")
    else:
        print("    stitch validation: no missing steps")

    eval_points = []

    for row in rows:
        metric_name, score = find_test_score(row)
        if metric_name is not None:
            eval_points.append((row["_step"], metric_name, score))

    if eval_points:
        print("    eval points:")
        for step, metric_name, score in eval_points:
            print(f"      step={step}: {metric_name}={score}")


def export_stitched(api, project, exp_name, segments):
    """
    Stitch exact W&B runs using explicit step windows.

    This is the right method for math-clean:
        uojla9ie:  0-299
        hkvxuwgl: 300-400
    """
    print(f"  {exp_name}: exact stitching {len(segments)} run segments")

    stitched_by_step = {}
    run_ids = []
    segment_meta = []

    for seg in segments:
        run_id = seg["run_id"]
        start = int(seg["start"])
        end = int(seg["end"])
        step_offset = int(seg.get("step_offset", 0))
        label = seg.get("label", run_id)

        run = get_run_by_id(api, project, run_id)
        run_ids.append(run.id)

        print(
            f"    fetching id={run.id} "
            f"name={run.name} "
            f"created={run.created_at} "
            f"segment={start}-{end} "
            f"offset={step_offset}"
        )

        segment_rows = export_run_segment(
            run=run,
            start=start,
            end=end,
            step_offset=step_offset,
            label=label,
        )

        for row in segment_rows:
            step = int(row["_step"])

            if step in stitched_by_step:
                old_src = stitched_by_step[step].get("_source_run_id")
                new_src = row.get("_source_run_id")
                raise ValueError(
                    f"Duplicate stitched step {step}: "
                    f"already from {old_src}, also from {new_src}. "
                    f"Fix segment windows."
                )

            stitched_by_step[step] = row

        segment_meta.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "start": start,
                "end": end,
                "step_offset": step_offset,
                "label": label,
                "n_unique_steps": len(segment_rows),
            }
        )

    stitched_rows = [stitched_by_step[s] for s in sorted(stitched_by_step)]

    validate_stitched_rows(stitched_rows, exp_name)

    return stitched_rows, run_ids, segment_meta


def export_merged(api, project, exp_name):
    """
    Generic merge for experiments where exact stitching is not needed.

    Strategy:
        - Sort runs by creation time, oldest first.
        - For each step, first run's row wins.
        - Later runs only fill missing keys.

    Do NOT use this for math-clean; use export_stitched instead.
    """
    runs = get_all_runs(api, project, exp_name)

    if not runs:
        print(f"  WARNING: no runs for {exp_name}")
        return [], []

    runs.sort(key=lambda r: r.created_at)

    print(f"  {exp_name}: generic merging {len(runs)} runs")

    run_ids = []

    for r in runs:
        steps = r.summary.get("_step", 0)
        print(f"    {r.name} id={r.id} steps={steps} created={r.created_at}")
        run_ids.append(r.id)

    all_rows_by_step = {}

    for run in runs:
        run_rows = export_run(run)

        for row in run_rows:
            step = row["_step"]

            if step not in all_rows_by_step:
                all_rows_by_step[step] = row
            else:
                existing = all_rows_by_step[step]
                for k, v in row.items():
                    if k not in existing:
                        existing[k] = v

    merged = [all_rows_by_step[s] for s in sorted(all_rows_by_step.keys())]

    print(f"    merged: {len(merged)} unique steps, {sum(len(r) for r in merged)} values")

    return merged, run_ids


def export_aime(api):
    """
    Export AIME summaries.

    Deduplicates by run name, keeping the latest run per name.
    """
    runs = list(api.runs(f"{ENTITY}/{AIME_PROJECT}"))

    best = {}

    for r in runs:
        name = r.name or r.id

        if name not in best or r.created_at > best[name].created_at:
            best[name] = r

    out = {}

    for name, run in best.items():
        summary = {}

        for k, v in run.summary.items():
            if not k.startswith("_") or k == "_runtime":
                summary[k] = json_safe(v)

        summary["run_id"] = run.id
        out[name] = summary

        print(
            f"  {name}: "
            f"pass@1={summary.get('pass@1', '?')} "
            f"pass@8={summary.get('pass@8', '?')}"
        )

    return out


def main():
    api = wandb.Api()

    results = {
        "training": {},
        "aime": {},
    }

    for project, names in TRAINING_PROJECTS.items():
        print(f"\n{project}")

        for exp_name in names:
            key = exp_name.replace("-", "_")

            if exp_name in STITCHED_EXPERIMENTS:
                rows, run_ids, segment_meta = export_stitched(
                    api=api,
                    project=project,
                    exp_name=exp_name,
                    segments=STITCHED_EXPERIMENTS[exp_name],
                )

                results["training"][key] = {
                    "run_ids": run_ids,
                    "experiment_name": exp_name,
                    "stitched": True,
                    "stitch_segments": segment_meta,
                    "rows": rows,
                }

            elif exp_name in MERGE_EXPERIMENTS:
                rows, run_ids = export_merged(api, project, exp_name)

                results["training"][key] = {
                    "run_ids": run_ids,
                    "experiment_name": exp_name,
                    "merged": True,
                    "rows": rows,
                }

            else:
                run = get_longest_run(api, project, exp_name)

                if not run:
                    continue

                data = export_run(run)

                results["training"][key] = {
                    "run_id": run.id,
                    "experiment_name": exp_name,
                    "rows": data,
                }

    print(f"\n{AIME_PROJECT}")
    results["aime"] = export_aime(api)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(
        f"\nDone. "
        f"Training: {len(results['training'])} runs, "
        f"AIME: {len(results['aime'])} evals"
    )

    for k, v in results["training"].items():
        if v.get("stitched"):
            flag = " stitched"
        elif v.get("merged"):
            flag = " merged"
        else:
            flag = ""

        print(f"  {k}: {len(v['rows'])} rows{flag}")


if __name__ == "__main__":
    main()
