"""
W&B -> results.json export (v3)
Pulls ALL history rows, filters client-side. No key-name guessing.

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
    ],
    "cs224r-trivia-hendrycks": [
        "math-clean",
        "math-trivia-only",
    ],
}

AIME_PROJECT = "cs224r-trivia-aime"

# We keep any key containing these substrings
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
    "_step",
]


def should_keep(key):
    for k in KEEP:
        if k in key:
            return True
    return False


def get_longest_run(api, project, exp_name):
    runs = list(api.runs(
        f"{ENTITY}/{project}",
        filters={"config.trainer.experiment_name": exp_name},
    ))
    if not runs:
        all_runs = list(api.runs(f"{ENTITY}/{project}"))
        runs = [r for r in all_runs if r.name == exp_name]
    if not runs:
        print(f"  WARNING: no runs for {exp_name}")
        return None
    best = max(runs, key=lambda r: r.summary.get("_step", 0))
    print(f"  {exp_name}: {best.name} id={best.id} steps={best.summary.get('_step',0)}")
    return best


def export_run(run):
    """Pull all history, filter client-side."""
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
            if not k.startswith("_"):
                try:
                    json.dumps(v)
                    summary[k] = v
                except (TypeError, ValueError):
                    summary[k] = str(v)
        summary["run_id"] = run.id
        out[name] = summary
        print(f"  {name}: pass@1={summary.get('pass@1','?')} pass@8={summary.get('pass@8','?')}")
    return out


def main():
    api = wandb.Api()
    results = {"training": {}, "aime": {}}

    for project, names in TRAINING_PROJECTS.items():
        print(f"\n{project}")
        for exp_name in names:
            run = get_longest_run(api, project, exp_name)
            if not run:
                continue
            data = export_run(run)
            key = exp_name.replace("-", "_")
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
        print(f"  {k}: {len(v['rows'])} rows")


if __name__ == "__main__":
    main()
