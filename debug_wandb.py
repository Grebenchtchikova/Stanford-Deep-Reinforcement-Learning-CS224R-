"""
Debug: inspect what's actually in the W&B runs
"""
import wandb

api = wandb.Api()
ENTITY = "greben-stanford-university"

# Check one training run
run = api.run(f"{ENTITY}/cs224r-trivia-gsm8k/q1e89jkn")
print(f"Run: {run.name} (id={run.id})")
print(f"State: {run.state}")
print()

# 1. Summary keys
print("=== SUMMARY KEYS ===")
for k, v in sorted(run.summary.items()):
    val_str = str(v)[:80] if v is not None else "None"
    print(f"  {k}: {val_str}")
print()

# 2. History: first 3 rows, all columns
print("=== HISTORY (first 3 rows via scan_history) ===")
count = 0
for row in run.scan_history(page_size=3):
    print(f"  Row {count}: {sorted(row.keys())}")
    count += 1
    if count >= 3:
        break

if count == 0:
    print("  [no history rows returned by scan_history]")

# Try alternative method regardless
print()
print("=== TRYING run.history() ===")
try:
    hist = run.history(samples=5)
    print(f"  Columns ({len(hist.columns)}): {list(hist.columns)[:30]}")
    if len(hist.columns) > 30:
        print(f"  ... and {len(hist.columns) - 30} more")
    print(f"  Shape: {hist.shape}")
except Exception as e:
    print(f"  Error: {e}")

print()

# 3. Check one AIME run
print("=== AIME RUN ===")
aime_run = api.run(f"{ENTITY}/cs224r-trivia-aime/i0gmssik")
print(f"Run: {aime_run.name} (id={aime_run.id})")
print("Summary keys:")
for k, v in sorted(aime_run.summary.items()):
    val_str = str(v)[:100] if v is not None else "None"
    print(f"  {k}: {val_str}")
