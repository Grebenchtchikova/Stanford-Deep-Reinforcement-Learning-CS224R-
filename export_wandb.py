"""
W&B -> results.json export (v4)
Pulls ALL history rows, filters client-side.
Merges multiple runs for the same experiment (e.g. math-clean original + resumed).

Usage:
    python export_wandb.py
"""
import wandb
import json

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

# Experiments that need merging across multiple runs
# For these, we pull ALL matching runs and stitch by step
MERGE_EXPERIMENTS = {"math-clean"}

AIME_PROJECT = "cs224r-trivia-aime"

KEEP = [
    "actor/entropy", "actor/kl_loss", "actor/ppo_kl",
    "actor/pg_clipfrac",
    "response_length/clip_ratio", "response_length/mean",
    "response_length/min", "response_length/max",
    "timing_s/testing", "timing_s/training", "timing_s/step",
    "timing_s/gen",
    "critic/score/mean", "critic/rewards/mean",
    "test_score", "reward/mean", "length/mean", "length/unknown",
    "perf/max_memory_allocated_gb",
    "steps/compute_matched_steps",
    "_step",
]


def should_keep(key):
    for k in KEEP:
        if k in key:
            return True
    return False


def get_all_runs(api, project, exp_name):
    """Get all runs matching an experiment name."""
    runs = list(api.runs(
        f"{ENTITY}/{project}",
        filters={"config.trainer.experiment_name": exp_name},
    ))
    if not runs:
        all_runs = list(api.runs(f"{ENTITY}/{project}"))
        runs = [r for r in all_runs if r.name == exp_name]
    return runs


def get_longest_run(api, project, exp_name):
    runs = get_all_runs(api, project, exp_name)
    if not runs:
        print(f"  WARNING: no runs for {exp_name}")
        return None
    best = max(runs, key=lambda r: r.summary.get("_step", 0))
    print(f"  {exp_name}: {best.name} id={best.id} steps={best.summary.get('_step', 0)}")
    return best


def export_run(run):
    """Pull all history from a single run, filter client-side."""
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


def export_merged(api, project, exp_name):
    """Pull all runs for an experiment name and merge by step.
    
    Strategy:
    - Sort runs by creation time (oldest first)
    - For each step, take the FIRST (original) run's data
    - Only use later runs to fill in steps not covered by earlier runs
    This ensures continuous original training takes priority.
    """
    runs = get_all_runs(api, project, exp_name)
    if not runs:
        print(f"  WARNING: no runs for {exp_name}")
        return [], []

    # Sort oldest first — original run takes priority
    runs.sort(key=lambda r: r.created_at)

    print(f"  {exp_name}: merging {len(runs)} runs:")
    run_ids = []
    for r in runs:
        steps = r.summary.get("_step", 0)
        print(f"    {r.name} id={r.id} steps={steps} created={r.created_at}")
        run_ids.append(r.id)

    # Pull all rows from all runs
    all_rows_by_step = {}  # step -> row (first writer wins)
    for run in runs:
        run_rows = export_run(run)
        for row in run_rows:
            step = row["_step"]
            if step not in all_rows_by_step:
                # First run to provide this step wins
                all_rows_by_step[step] = row
            else:
                # Merge: fill in keys the existing row doesn't have
                existing = all_rows_by_step[step]
                for k, v in row.items():
                    if k not in existing:
                        existing[k] = v

    # Sort by step
    merged = [all_rows_by_step[s] for s in sorted(all_rows_by_step.keys())]
    print(f"    Merged: {len(merged)} unique steps, {sum(len(r) for r in merged)} values")
    return merged, run_ids


def export_aime(api):
    runs = list(api.runs(f"{ENTITY}/{AIME_PROJECT}"))
    # Dedup: latest per name
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
                try:
                    json.dumps(v)
                    summary[k] = v
                except (TypeError, ValueError):
                    summary[k] = str(v)
        summary["run_id"] = run.id
        out[name] = summary
        print(f"  {name}: pass@1={summary.get('pass@1', '?')} pass@8={summary.get('pass@8', '?')}")
    return out


def main():
    api = wandb.Api()
    results = {"training": {}, "aime": {}}

    for project, names in TRAINING_PROJECTS.items():
        print(f"\n{project}")
        for exp_name in names:
            key = exp_name.replace("-", "_")

            if exp_name in MERGE_EXPERIMENTS:
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

    print(f"\nDone. Training: {len(results['training'])} runs, AIME: {len(results['aime'])} evals")
    for k, v in results["training"].items():
        merged = " (merged)" if v.get("merged") else ""
        print(f"  {k}: {len(v['rows'])} rows{merged}")


if __name__ == "__main__":
    main()
