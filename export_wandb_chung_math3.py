"""
W&B -> results_chung.json — ONLY Chung's three MATH runs.

Entity:  cfw20-stanford-university
Project: rlad-hendrycks
Runs:    qwen3-1p7b-hendrycks-grpo-clean    (original)
         qwen3-1p7b-hendrycks-grpo-trivia   (trivia)
         qwen3-1p7b-hendrycks-grpo-mixed    (mixed)

Output schema matches your version-G export (same KEEP list, same row format),
so results_chung.json drops straight into analyze_results.py.

NOTE on labels: the run names say "grpo"; Chung confirmed these are the e3 runs,
so the output keys say e3 (chung_math_e3_*). Each entry still stores the real W&B
run name and experiment_name, so the provenance is intact if you ever want to
double-check. Worth a glance at one run's Config for the e3 markers (negative
gradient / asymmetric clip ~0.5 / curriculum) just to have it on record.

SETUP (once): log in as Chung so wandb.Api() reads his account.
    wandb login            # paste Chung's key at the prompt
    # switch back afterwards with:  wandb login --relogin

USAGE:
    python export_wandb_chung_math3.py
"""

import json
import wandb


ENTITY = "cfw20-stanford-university"
PROJECT = "rlad-hendrycks"
OUTPUT_FILE = "results_chung.json"

# run name (as shown in W&B)  ->  (output key, variant label)
WANTED = {
    "qwen3-1p7b-hendrycks-grpo-clean":  ("chung_math_e3_clean",  "original"),
    "qwen3-1p7b-hendrycks-grpo-trivia": ("chung_math_e3_trivia", "trivia"),
    "qwen3-1p7b-hendrycks-grpo-mixed":  ("chung_math_e3_mixed",  "mixed"),
}

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
    return any(k in key for k in KEEP)


def experiment_name_of(run):
    cfg = run.config or {}
    trainer = cfg.get("trainer")
    if isinstance(trainer, dict) and trainer.get("experiment_name"):
        return trainer["experiment_name"]
    return cfg.get("trainer.experiment_name")


def export_run(run):
    """Pull all history rows, filter client-side to KEEP."""
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


def main():
    api = wandb.Api()

    # The project has only a handful of runs — list them and match the three we want
    # by run name (what you can see), with experiment_name as a secondary match.
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"{ENTITY}/{PROJECT}: {len(all_runs)} run(s) present")

    # group candidate runs per wanted name (handles a crashed+resumed duplicate)
    matches = {name: [] for name in WANTED}
    for r in all_runs:
        exp = experiment_name_of(r)
        for name in WANTED:
            if r.name == name or exp == name:
                matches[name].append(r)

    results = {"training": {}, "aime": {}}

    for name, (key, variant) in WANTED.items():
        runs = matches[name]
        if not runs:
            print(f"  !! '{name}' not found — check the exact spelling in W&B")
            continue
        if len(runs) > 1:
            print(
                f"  NOTE: '{name}' has {len(runs)} runs (crashed/resumed?) — taking longest"
            )
        run = max(runs, key=lambda r: r.summary.get("_step", 0) or 0)
        print(
            f"  {variant:9s} <- {run.name} id={run.id} "
            f"steps={run.summary.get('_step', 0)}"
        )
        rows = export_run(run)
        results["training"][key] = {
            "run_id": run.id,
            "run_name": run.name,
            "experiment_name": experiment_name_of(run),
            "project": PROJECT,
            "variant": variant,
            "rows": rows,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {OUTPUT_FILE}  ({len(results['training'])}/3 runs)")
    for key, v in results["training"].items():
        print(f"  {key} [{v['variant']}]: {len(v['rows'])} rows")
    for name, (key, variant) in WANTED.items():
        if key not in results["training"]:
            print(f"  MISSING: {variant} ({name})")


if __name__ == "__main__":
    main()
