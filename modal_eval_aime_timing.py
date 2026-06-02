#!/usr/bin/env python3
"""
AIME timing eval — identical to modal_eval_aime.py but:
  - Logs to separate W&B project (cs224r-trivia-aime-timing)
  - Measures and logs generation wall time
  - Measures response lengths

Usage (open 5 terminals, fire simultaneously):
    modal run modal_eval_aime_timing.py --model track-a-gsm8k
    modal run modal_eval_aime_timing.py --model track-b-gsm8k
    modal run modal_eval_aime_timing.py --model track-c-gsm8k
    modal run modal_eval_aime_timing.py --model track-d-gsm8k
    modal run modal_eval_aime_timing.py --model track-e-gsm8k
"""

import modal
import time

app = modal.App("cs224r-aime-timing")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir("../cs224r-project-e3", "/root/verl", copy=True)
    .run_commands("pip install -e /root/verl", "pip install seaborn")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

WANDB_PROJECT = "cs224r-trivia-aime-timing"
DATA_DIR = "/data/eval/aime_timing"

MODEL_CONFIG = {
    "track-a-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-clean_hf",
        "wandb_name": "Track A GSM8K",
    },
    "track-b-gsm8k": {
        "path": "/data/ckpts/gsm8k/gsm8k-trivia-only_hf",
        "wandb_name": "Track B GSM8K",
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

AIME_HF_ID = "CMU-AIRe/hmmt-aime-2025"
AIME_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
N_SAMPLES = 8
MAX_RESPONSE_LENGTH = 8192


def _pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    import numpy as np
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=6 * 3600,
)
def run_eval(model: str):
    import os
    import subprocess
    import json
    import pandas as pd
    import numpy as np
    from datasets import load_dataset
    import wandb

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(DATA_DIR, exist_ok=True)

    config = MODEL_CONFIG[model]
    model_path = config["path"]
    wandb_name = config["wandb_name"]

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ---- W&B ----
    wandb.init(
        project=WANDB_PROJECT,
        name=wandb_name,
        config={
            "model": model,
            "model_path": model_path,
            "n_samples": N_SAMPLES,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "gpu": "A100-80GB",
        },
    )

    # ---- Prepare AIME data ----
    data_path = os.path.join(DATA_DIR, "aime_timing.parquet")
    if not os.path.exists(data_path):
        ds = load_dataset(AIME_HF_ID)
        split = "test" if "test" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        for c in ["source", "competition", "dataset", "origin"]:
            if c in df.columns:
                mask = df[c].astype(str).str.lower().str.contains("aime")
                df = df[mask].reset_index(drop=True)
                break
        prob_col = next(c for c in ["problem", "question", "prompt"] if c in df.columns)
        ans_col = next(c for c in ["answer", "solution", "ground_truth"] if c in df.columns)
        rows = []
        for idx, row in df.iterrows():
            rows.append({
                "data_source": "aime",
                "prompt": [{"role": "user", "content": f"{row[prob_col]} {AIME_INSTRUCTION}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(row[ans_col])},
                "extra_info": {"split": split, "index": int(idx)},
            })
        pd.DataFrame(rows).to_parquet(data_path, index=False)
        print(f"[data] Wrote {len(rows)} AIME problems")

    # ---- Generate (timed) ----
    output_path = os.path.join(DATA_DIR, f"{model}_timing_outputs.parquet")

    cmd = [
        "python3", "-m", "verl.trainer.main_generation",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"data.path={data_path}",
        "data.prompt_key=prompt",
        f"data.n_samples={N_SAMPLES}",
        "data.batch_size=24",
        f"data.output_path={output_path}",
        f"model.path={model_path}",
        "+model.trust_remote_code=True",
        "rollout.temperature=0.6",
        "rollout.top_k=20",
        "rollout.top_p=0.95",
        "rollout.do_sample=True",
        "rollout.prompt_length=1024",
        f"rollout.response_length={MAX_RESPONSE_LENGTH}",
        "rollout.tensor_model_parallel_size=1",
        "rollout.gpu_memory_utilization=0.9",
        "rollout.enforce_eager=False",
        "rollout.free_cache_engine=False",
        "rollout.max_num_batched_tokens=50000",
        "+rollout.extrapolation_val=False",
        "+rollout.extrapolation_length=0",
    ]

    print(f"\n[{wandb_name}] Starting generation...")
    gen_start = time.time()
    subprocess.run(cmd, check=True, cwd="/root/verl")
    gen_elapsed = time.time() - gen_start
    print(f"[{wandb_name}] Generation done: {gen_elapsed/60:.1f} min")

    # ---- Measure response lengths ----
    out_df = pd.read_parquet(output_path)
    all_lengths = []
    for responses in out_df["responses"]:
        for r in responses:
            all_lengths.append(len(r))
    total_chars = sum(all_lengths)
    mean_chars = np.mean(all_lengths)

    # ---- Score accuracy ----
    print(f"[{wandb_name}] Scoring...")
    from verl.utils.reward_score.curriculum_math.compute_score import compute_score

    num_problems = len(out_df)
    correctness = np.zeros((num_problems, N_SAMPLES), dtype=np.int32)
    for i, row in out_df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        for j, resp in enumerate(row["responses"]):
            try:
                score = float(compute_score(
                    data_source="aime", solution_str=resp,
                    ground_truth=gt, extra_info=None,
                ))
            except Exception:
                score = 0.0
            correctness[i, j] = int(score == 1.0)

    pass1 = float(correctness.mean())
    pass4 = float(np.mean([_pass_at_k(N_SAMPLES, int(correctness[i].sum()), 4) for i in range(num_problems)]))
    pass8 = float(np.mean([_pass_at_k(N_SAMPLES, int(correctness[i].sum()), 8) for i in range(num_problems)]))

    # ---- Results ----
    result = {
        "model": model,
        "generation_time_s": round(gen_elapsed, 1),
        "generation_time_min": round(gen_elapsed / 60, 1),
        "total_chars": total_chars,
        "mean_chars_per_response": round(mean_chars, 0),
        "num_responses": len(all_lengths),
        "chars_per_second": round(total_chars / gen_elapsed, 0),
        "pass@1": round(pass1, 4),
        "pass@4": round(pass4, 4),
        "pass@8": round(pass8, 4),
    }

    wandb.log(result)
    wandb.finish()

    result_path = os.path.join(DATA_DIR, f"{model}_timing.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    print(f"\n{'='*60}")
    print(f"  {wandb_name}")
    print(f"  Generation: {gen_elapsed/60:.1f} min")
    print(f"  Throughput: {total_chars/gen_elapsed:,.0f} chars/s")
    print(f"  Mean response: {mean_chars:,.0f} chars")
    print(f"  pass@1={pass1:.3f}  pass@4={pass4:.3f}  pass@8={pass8:.3f}")
    print(f"{'='*60}")

    return result


@app.local_entrypoint()
def main(model: str = "track-a-gsm8k"):
    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown --model={model!r}; choose from {list(MODEL_CONFIG)}")
    print(f"[main] Launching {MODEL_CONFIG[model]['wandb_name']}...")
    result = run_eval.remote(model=model)
    print(f"\n{result}")
