"""
W&B → results.json export script
Selects the longest run per experiment name, exports training curves + AIME evals.

Usage:
    pip install wandb
    python export_wandb.py

Output: results.json
"""

import wandb
import json
from collections import defaultdict

# ---- CONFIG ----
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

# Metrics to pull from training runs (all logged per step unless noted)
TRAINING_METRICS = [
    "global_step",
    # Validation (sparse — logged at test_freq intervals)
    "val/openai/gsm8k/test_score/",
    "val/openai/gsm8k/test_score/unknown",
    "val/openai/gsm8k/reward/mean",
    "val/openai/gsm8k/length/mean",
    "val/openai/gsm8k/length/unknown",
    "val/DigitalLearningGmbH/MATH-lighteval/test_score/",
    "val/DigitalLearningGmbH/MATH-lighteval/test_score/unknown",
    "val/DigitalLearningGmbH/MATH-lighteval/reward/mean",
    "val/DigitalLearningGmbH/MATH-lighteval/length/mean",
    "val/DigitalLearningGmbH/MATH-lighteval/length/unknown",
    # Training (dense — logged every step)
    "actor/entropy",
    "response_length/mean",
    "response_length/min",
    "response_length/clip_ratio",
    "timing_s/training",
    "timing_s/testing",
]

# AIME metrics
AIME_METRICS = [
    "pass_at_1",
    "pass_at_1_mean",
    "pass_at_4",
    "pass_at_8",
    "n_samples",
    "num_problems",
]


def get_longest_run(api, project, experiment_name):
    """Find the run with the most logged steps for a given experiment name."""
    runs = api.runs(
        f"{ENTITY}/{project}",
        filters={"config.trainer.experiment_name": experiment_name},
    )
    if not runs:
        # Try display name fallback
        runs = api.runs(
            f"{ENTITY}/{project}",
            filters={"display_name": experiment_name},
        )
    if not runs:
        print(f"  WARNING: No runs found for {experiment_name} in {project}")
        return None

    # Pick the run with the most history rows
    best_run = None
    best_steps = -1
    for run in runs:
        last_step = run.summary.get("_step", 0)
        if last_step > best_steps:
            best_steps = last_step
            best_run = run
    
    if best_run:
        print(f"  Selected: {best_run.name} (id={best_run.id}, {best_steps} steps)")
    return best_run


def export_training_run(run):
    """Export all training metrics from a run."""
    history = run.scan_history(keys=TRAINING_METRICS, page_size=1000)
    
    data = defaultdict(list)
    for row in history:
        for key in TRAINING_METRICS:
            if key in row and row[key] is not None:
                # Use global_step as x-axis; fall back to _step
                step = row.get("global_step", row.get("_step", 0))
                data[key].append({"step": step, "value": row[key]})
    
    return dict(data)


def export_aime_runs(api):
    """Export all AIME evaluation runs."""
    runs = api.runs(f"{ENTITY}/{AIME_PROJECT}")
    aime_data = {}
    
    for run in runs:
        name = run.name or run.display_name or run.id
        print(f"  AIME run: {name} (id={run.id})")
        
        summary = {}
        for key in AIME_METRICS:
            val = run.summary.get(key)
            if val is not None:
                summary[key] = val
        
        # Also grab per-problem scores if logged as a table
        per_problem = {}
        try:
            for key, val in run.summary.items():
                if "problem" in key.lower() or "score" in key.lower():
                    per_problem[key] = val
        except Exception:
            pass
        
        if per_problem:
            summary["per_problem_raw"] = per_problem
        
        # Try to get config to identify which model this evaluated
        model_config = run.config.get("model", name)
        summary["model_config"] = model_config
        summary["run_name"] = name
        
        aime_data[name] = summary
    
    return aime_data


def main():
    api = wandb.Api()
    results = {
        "training_curves": {},
        "aime": {},
        "metadata": {
            "entity": ENTITY,
            "export_note": "Longest run per experiment name auto-selected",
        },
    }
    
    # ---- Training runs ----
    for project, experiment_names in TRAINING_PROJECTS.items():
        print(f"\nProject: {project}")
        for exp_name in experiment_names:
            print(f"  Experiment: {exp_name}")
            run = get_longest_run(api, project, exp_name)
            if run is None:
                continue
            
            data = export_training_run(run)
            
            # Store with a clean key name
            key = exp_name.replace("-", "_")
            results["training_curves"][key] = {
                "run_id": run.id,
                "run_name": run.name,
                "project": project,
                "experiment_name": exp_name,
                "total_steps": run.summary.get("_step", 0),
                "metrics": data,
            }
    
    # ---- AIME evals ----
    print(f"\nProject: {AIME_PROJECT}")
    results["aime"] = export_aime_runs(api)
    
    # ---- Save ----
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Exported to {output_path}")
    print(f"Training runs: {len(results['training_curves'])}")
    for key, val in results["training_curves"].items():
        n_metrics = sum(len(v) for v in val["metrics"].values())
        print(f"  {key}: {n_metrics} data points across {len(val['metrics'])} metrics")
    print(f"AIME evals: {len(results['aime'])}")
    for key, val in results["aime"].items():
        print(f"  {key}: {list(val.keys())}")


if __name__ == "__main__":
    main()
