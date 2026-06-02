"""
Pull _runtime from all W&B runs across all projects.
Quick utility — doesn't modify results.json.

Usage:
    python fetch_runtimes.py
"""
import wandb

ENTITY = "greben-stanford-university"

PROJECTS = [
    "cs224r-trivia-gsm8k",
    "cs224r-trivia-hendrycks",
    "cs224r-trivia-aime",
    "cs224r-trivia-aime-timing",
]

api = wandb.Api()

for project in PROJECTS:
    print(f"\n{'='*60}")
    print(f"  {project}")
    print(f"{'='*60}")
    print(f"  {'Name':<45} {'Runtime':>10} {'Steps':>8}")
    print(f"  {'-'*65}")

    try:
        runs = list(api.runs(f"{ENTITY}/{project}"))
    except Exception as e:
        print(f"  Error: {e}")
        continue

    for r in sorted(runs, key=lambda x: x.name or ""):
        name = r.name or r.id
        runtime = r.summary.get("_runtime", None)
        steps = r.summary.get("_step", None)

        if runtime is not None:
            hours = runtime / 3600
            mins = runtime / 60
            if hours >= 1:
                rt_str = f"{hours:.1f}h"
            else:
                rt_str = f"{mins:.1f}m"
        else:
            rt_str = "?"

        steps_str = str(steps) if steps is not None else "?"
        print(f"  {name:<45} {rt_str:>10} {steps_str:>8}")
