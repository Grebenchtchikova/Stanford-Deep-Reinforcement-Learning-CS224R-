#!/usr/bin/env python3
"""
AIME evaluation for all trained checkpoints.

Evaluates GSM8K-trained and MATH-trained models on AIME 2025,
with clear W&B labels for each run.

Usage:
    # GSM8K-trained models on AIME
    modal run modal_eval_aime.py --model track-a-gsm8k
    modal run modal_eval_aime.py --model track-b-gsm8k

    # MATH-trained models on AIME (run after MATH training completes)
    modal run modal_eval_aime.py --model track-a-math
    modal run modal_eval_aime.py --model track-b-math

    # All four at once (detached)
    modal run --detach modal_eval_aime.py --model track-a-gsm8k
    modal run --detach modal_eval_aime.py --model track-b-gsm8k
    modal run --detach modal_eval_aime.py --model track-a-math
    modal run --detach modal_eval_aime.py --model track-b-math
"""

import modal

app = modal.App("cs224r-trivia-eval-aime")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir("../cs224r-project-e3", "/root/verl", copy=True)
    .run_commands("pip install -e /root/verl", "pip install seaborn")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/eval/aime"

MODEL_CONFIG = {
    "track-a-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-clean_hf",
        "wandb_name": "Track A GSM8K",
    },
    "track-b-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-trivia-only_hf",
        "wandb_name": "Track B GSM8K",
    },
    "track-a-math": {
        "path": "/data/ckpts/math/math-clean_hf",
        "wandb_name": "Track A MATH",
    },
    "track-b-math": {
        "path": "/data/ckpts/math/math-trivia-only_hf",
        "wandb_name": "Track B MATH",
    },
    "track-c-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-partial-e3-clean_hf",
        "wandb_name": "Track C GSM8K (partial e3 clean)",
    },
    "track-d-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-partial-e3-trivia_hf",
        "wandb_name": "Track D GSM8K (partial e3 trivia)",
    },
    "track-e-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-track-e-2M-partial-e3_hf",
        "wandb_name": "Track E GSM8K (2M mixed partial e3)",
    },
}

# AIME dataset configuration
AIME_HF_ID = "CMU-AIRe/hmmt-aime-2025"
AIME_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _extract_answer(raw):
    """AIME answers are raw integers."""
    return str(raw)


def _prepare_aime_data(num_problems, tag):
    """Download AIME from HuggingFace and format as parquet."""
    import os
    import pandas as pd
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"aime_{tag}.parquet")

    ds = load_dataset(AIME_HF_ID)
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    print(f"[data] Using split '{split_name}' with {len(ds[split_name])} rows")
    df = ds[split_name].to_pandas()
    print(f"[data] Columns: {list(df.columns)}")

    # Filter to AIME problems if competition column exists
    for c in ["source", "competition", "dataset", "origin"]:
        if c in df.columns:
            mask = df[c].astype(str).str.lower().str.contains("aime")
            print(f"[data] Filtering by {c} contains 'aime': {int(mask.sum())}/{len(df)} rows kept")
            df = df[mask].reset_index(drop=True)
            break

    # Find problem and answer columns
    prob_col = next((c for c in ["problem", "question", "prompt"] if c in df.columns), None)
    ans_col = next((c for c in ["answer", "solution", "ground_truth"] if c in df.columns), None)
    if prob_col is None or ans_col is None:
        raise ValueError(f"Could not find question/answer columns. Have: {list(df.columns)}")

    if num_problems is not None:
        df = df.head(num_problems).reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        question = f"{row[prob_col]} {AIME_INSTRUCTION}"
        gt = _extract_answer(row[ans_col])
        rows.append({
            "data_source": "aime",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(gt)},
            "extra_info": {"split": split_name, "index": int(idx)},
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[data] Wrote {len(out_df)} AIME prompts -> {out_path}")
    return out_path, len(out_df)


def _run_generation(data_path, output_path, model_id, n_samples, max_response_length):
    """Run verl generation on AIME problems."""
    import subprocess

    batch_size = 24
    cmd = [
        "python3", "-m", "verl.trainer.main_generation",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"data.path={data_path}",
        "data.prompt_key=prompt",
        f"data.n_samples={n_samples}",
        f"data.batch_size={batch_size}",
        f"data.output_path={output_path}",
        f"model.path={model_id}",
        "+model.trust_remote_code=True",
        "rollout.temperature=0.6",
        "rollout.top_k=20",
        "rollout.top_p=0.95",
        "rollout.do_sample=True",
        "rollout.prompt_length=1024",
        f"rollout.response_length={max_response_length}",
        "rollout.tensor_model_parallel_size=1",
        "rollout.gpu_memory_utilization=0.9",
        "rollout.enforce_eager=False",
        "rollout.free_cache_engine=False",
        "rollout.max_num_batched_tokens=50000",
        "+rollout.extrapolation_val=False",
        "+rollout.extrapolation_length=0",
    ]
    print("[gen] Running:")
    print("    " + " \\\n    ".join(cmd))
    subprocess.run(cmd, check=True)


def _pass_at_k(n, c, k):
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    import numpy as np
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _score_outputs(output_path, n_samples, model_tag):
    """Score AIME outputs using curriculum_math scorer."""
    import os
    import json
    import pandas as pd
    import numpy as np
    from verl.utils.reward_score.curriculum_math.compute_score import compute_score

    df = pd.read_parquet(output_path)
    num_problems = len(df)
    print(f"[score] Loaded {num_problems} problems from {output_path}")

    correctness = np.zeros((num_problems, n_samples), dtype=np.int32)
    extract_failures = 0
    per_problem_rows = []

    for i, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        responses = row["responses"]
        for j, resp in enumerate(responses):
            try:
                score = float(compute_score(
                    data_source="aime",
                    solution_str=resp,
                    ground_truth=gt,
                    extra_info=None,
                ))
            except Exception as e:
                print(f"[score] Error on problem {i} sample {j}: {e}")
                score = 0.0
            if score == 0.0 and "\\boxed" not in resp:
                extract_failures += 1
            correctness[i, j] = int(score == 1.0)

        per_problem_rows.append({
            "problem_idx": int(i),
            "ground_truth": str(gt),
            "n_samples": n_samples,
            "n_correct": int(correctness[i].sum()),
            "accuracy": float(correctness[i].mean()),
        })

    metrics = {
        "dataset": "aime",
        "model": model_tag,
        "scorer": "curriculum_math",
        "num_problems": int(num_problems),
        "n_samples": int(n_samples),
        "extract_failures": int(extract_failures),
        "pass@1_mean": float(correctness.mean()),
    }
    for k in (1, 4, 8, 16):
        if k <= n_samples:
            vals = [_pass_at_k(n_samples, int(correctness[i].sum()), k) for i in range(num_problems)]
            metrics[f"pass@{k}"] = float(np.mean(vals))

    metrics_path = os.path.join(DATA_DIR, f"metrics_aime_{model_tag}.json")
    per_problem_path = os.path.join(DATA_DIR, f"per_problem_aime_{model_tag}.csv")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(per_problem_rows).to_csv(per_problem_path, index=False)

    print("\n=== AIME Eval Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[score] metrics  -> {metrics_path}")
    print(f"[score] per-prob -> {per_problem_path}")

    return metrics


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_eval(model: str, n_samples: int, num_problems, max_response_length: int):
    import os
    import wandb

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown --model={model!r}; choose from {list(MODEL_CONFIG)}")

    config = MODEL_CONFIG[model]
    model_path = config["path"]
    wandb_name = config["wandb_name"]

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. "
            f"Run modal_convert_ckpt.py first."
        )

    # Initialize W&B with clear labeling
    wandb.init(
        project="cs224r-trivia-aime",
        name=wandb_name,
        config={
            "model": model,
            "model_path": model_path,
            "n_samples": n_samples,
            "max_response_length": max_response_length,
            "dataset": "aime",
        },
    )

    # 1. Prepare data
    tag = f"{model}_n{n_samples}"
    data_path, n_problems = _prepare_aime_data(
        num_problems=num_problems,
        tag=tag,
    )

    # 2. Generate
    output_path = os.path.join(DATA_DIR, f"aime_{model}_{tag}_outputs.parquet")
    _run_generation(
        data_path=data_path,
        output_path=output_path,
        model_id=model_path,
        n_samples=n_samples,
        max_response_length=max_response_length,
    )

    # 3. Score
    metrics = _score_outputs(
        output_path=output_path,
        n_samples=n_samples,
        model_tag=model,
    )

    # Log to W&B
    wandb.log(metrics)
    wandb.finish()

    vol.commit()
    return metrics


@app.local_entrypoint()
def main(
    model: str = "track-a-gsm8k",
    n_samples: int = 8,
    num_problems: int = -1,
    max_response_length: int = 8192,
):
    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown --model={model!r}; choose from {list(MODEL_CONFIG)}")

    num_problems_v = None if num_problems < 0 else int(num_problems)

    print(f"[main] model={model} n_samples={n_samples} "
          f"max_response_length={max_response_length} "
          f"num_problems={num_problems_v}")

    metrics = run_eval.remote(
        model=model,
        n_samples=n_samples,
        num_problems=num_problems_v,
        max_response_length=max_response_length,
    )
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
