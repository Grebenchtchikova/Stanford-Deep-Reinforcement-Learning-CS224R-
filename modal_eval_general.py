#!/usr/bin/env python3
"""
Unified Modal evaluation: run verl main_generation on AIME / MATH / GSM8K
and score with the appropriate scorer per dataset.

Usage:
    # GSM8K with base Qwen model
    modal run modal_eval_general.py --dataset gsm8k --model qwen

    # GSM8K with trained Track A checkpoint
    modal run modal_eval_general.py --dataset gsm8k --model track_a

    # GSM8K with trained Track B checkpoint
    modal run modal_eval_general.py --dataset gsm8k --model track_b

    # GSM8K with e3 published checkpoint
    modal run modal_eval_general.py --dataset gsm8k --model e3

    # MATH-500 with Track A
    modal run modal_eval_general.py --dataset math --model track_a

    # Smoke tests
    modal run modal_eval_general.py --dataset gsm8k --num-problems 5 --n-samples 1 --output-tag smoke
    modal run modal_eval_general.py --dataset math  --num-problems 5 --n-samples 1 --max-response-length 2048 --output-tag smoke
    modal run modal_eval_general.py --dataset aime  --num-problems 2 --n-samples 2 --max-response-length 2048 --output-tag smoke

Scorer choices:
  - AIME  -> curriculum_math (SymPy equivalence)
  - MATH  -> curriculum_math (SymPy equivalence handles equivalent answer forms)
  - GSM8K -> gsm8k.compute_score with method='flexible' (finds last number in response)
"""

import modal

app = modal.App("cs224r-trivia-eval")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .run_commands("pip install --no-deps verl==0.2.0.post2", "pip install seaborn lm-eval")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/eval"

MODEL_IDS = {
    "qwen": {
        "gsm8k": "Qwen/Qwen3-1.7B",
        "math": "Qwen/Qwen3-1.7B",
        "aime": "Qwen/Qwen3-1.7B",
    },
    "e3": {
        "gsm8k": "CMU-AIRe/e3-1.7B",
        "math": "CMU-AIRe/e3-1.7B",
        "aime": "CMU-AIRe/e3-1.7B",
    },
    "track_a": {
        "gsm8k": "/data/ckpts/gsm8k/gsm8k-clean_hf",
        "math": "/data/ckpts/math/math-clean_hf",
    },
    "track_b": {
        "gsm8k": "/data/ckpts/gsm8k/gsm8k-trivia-only_hf",
        "math": "/data/ckpts/math/math-trivia-only_hf",
    },
}

# Per-dataset configuration
DATASETS = {
    "aime": {
        "hf_id":         "CMU-AIRe/hmmt-aime-2025",
        "hf_config":     None,
        "default_split": "test",
        "competition":   "aime",
        "question_cols": ["problem", "question", "prompt"],
        "answer_cols":   ["answer", "solution", "ground_truth"],
        "answer_extractor": "raw",
        "data_source":   "aime",
        "instruction":   "Let's think step by step and output the final answer within \\boxed{}.",
        "default_n_samples":       16,
        "default_response_length": 32768,
        "scorer":        "curriculum_math",
    },
    "math": {
        "hf_id":         "HuggingFaceH4/MATH-500",
        "hf_config":     None,
        "default_split": "test",
        "competition":   None,
        "question_cols": ["problem"],
        "answer_cols":   ["solution"],
        "answer_extractor": "boxed",
        "data_source":   "HuggingFaceH4/MATH-500",
        "instruction":   "Let's think step by step and output the final answer within \\boxed{}.",
        "default_n_samples":       1,
        "default_response_length": 2048,
        "scorer":        "math",
    },
    "gsm8k": {
        "hf_id":         "openai/gsm8k",
        "hf_config":     "main",
        "default_split": "test",
        "competition":   None,
        "question_cols": ["question"],
        "answer_cols":   ["answer"],
        "answer_extractor": "hash",
        "data_source":   "openai/gsm8k",
        "instruction":   'Let\'s think step by step and output the final answer after "####".',
        "default_n_samples":       4,
        "default_response_length": 32768,
        "scorer":        "gsm8k_flexible",
    },
}


# ----------------------------- helpers -----------------------------

def _extract_answer(raw, extractor):
    """Pull the ground-truth answer out of a dataset's answer column."""
    s = str(raw)
    if extractor == "raw":
        return s
    if extractor == "boxed":
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        boxed = last_boxed_only_string(s)
        if boxed is None:
            return s
        try:
            return remove_boxed(boxed)
        except Exception:
            return s
    if extractor == "hash":
        import re
        m = re.search(r"#### (\-?[0-9\.\,]+)", s)
        if m is None:
            return s
        return m.group(1).replace(",", "").replace("$", "").strip()
    raise ValueError(f"Unknown answer extractor: {extractor}")


def _apply_math_subset(df, subset):
    """Apply MATH-specific subset filtering."""
    if subset == "all":
        return df
    if subset == "500":
        return df.reset_index(drop=True)
    if subset == "level5":
        if "level" not in df.columns:
            print(f"[data] WARNING: --subset level5 requested but no 'level' column; keeping all rows")
            return df
        mask = df["level"].astype(str).str.contains("5")
        print(f"[data] level5 filter: {int(mask.sum())}/{len(df)} rows kept")
        return df[mask].reset_index(drop=True)
    raise ValueError(f"Unknown --subset: {subset}")


def _prepare_dataset(dataset_key, subset, split, num_problems, tag):
    """Download HF dataset, filter, format as RLHF parquet matching verl's schema."""
    import os
    import pandas as pd
    from datasets import load_dataset

    cfg = DATASETS[dataset_key]
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{dataset_key}_{tag}.parquet")

    if cfg["hf_config"] is not None:
        ds = load_dataset(cfg["hf_id"], cfg["hf_config"])
    else:
        ds = load_dataset(cfg["hf_id"])

    if split in ds:
        split_name = split
    else:
        split_name = list(ds.keys())[0]
        print(f"[data] Split '{split}' not found; falling back to '{split_name}'")
    print(f"[data] Using split '{split_name}' with {len(ds[split_name])} rows")
    ds = ds[split_name]

    df = ds.to_pandas()
    print(f"[data] Columns: {list(df.columns)}")

    # AIME: filter by competition column if present
    if cfg["competition"] is not None:
        comp_col = None
        for c in ["source", "competition", "dataset", "origin"]:
            if c in df.columns:
                comp_col = c
                break
        if comp_col is not None:
            wanted = cfg["competition"].lower()
            mask = df[comp_col].astype(str).str.lower().str.contains(wanted)
            print(
                f"[data] Filtering by {comp_col} contains '{wanted}': "
                f"{int(mask.sum())}/{len(df)} rows kept"
            )
            df = df[mask].reset_index(drop=True)

    # MATH: subset filtering
    if dataset_key == "math":
        df = _apply_math_subset(df, subset)

    # Locate problem + answer columns
    prob_col = next((c for c in cfg["question_cols"] if c in df.columns), None)
    ans_col = next((c for c in cfg["answer_cols"] if c in df.columns), None)
    if prob_col is None or ans_col is None:
        raise ValueError(
            f"Could not find question/answer columns. "
            f"Have: {list(df.columns)}; want one of question={cfg['question_cols']} "
            f"answer={cfg['answer_cols']}"
        )
    print(f"[data] problem_col={prob_col}, answer_col={ans_col}, extractor={cfg['answer_extractor']}")

    if num_problems is not None:
        df = df.head(num_problems).reset_index(drop=True)
        print(f"[data] Truncated to {len(df)} problems")

    rows = []
    for idx, row in df.iterrows():
        question = f"{row[prob_col]} {cfg['instruction']}"
        gt = _extract_answer(row[ans_col], cfg["answer_extractor"])
        extra = {"split": split_name, "index": int(idx)}
        for opt in ("level", "type", "subject"):
            if opt in df.columns:
                extra[opt] = str(row[opt])
        rows.append(
            {
                "data_source": cfg["data_source"],
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(gt)},
                "extra_info": extra,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[data] Wrote {len(out_df)} prompts -> {out_path}")
    return out_path, len(out_df)


def _run_generation(
    data_path,
    output_path,
    model_id,
    n_samples,
    batch_size,
    max_prompt_length,
    max_response_length,
):
    """Invoke verl.trainer.main_generation as a subprocess."""
    import subprocess

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
        f"rollout.prompt_length={max_prompt_length}",
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
    """Unbiased pass@k estimator from Chen et al. (HumanEval)."""
    if n - c < k:
        return 1.0
    import numpy as np
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _get_score_fn(cfg):
    """Return (score_fn, extract_check_fn) pair for the dataset's configured scorer."""
    scorer = cfg["scorer"]
    data_source = cfg["data_source"]

    if scorer == "curriculum_math":
        from verl.utils.reward_score.curriculum_math.compute_score import compute_score

        def score_fn(resp, gt):
            return float(compute_score(
                data_source=data_source,
                solution_str=resp,
                ground_truth=gt,
                extra_info=None,
            ))

        def extract_check(resp):
            return "\\boxed" in resp

        return score_fn, extract_check

    if scorer == "math":
        from verl.utils.reward_score.math import compute_score

        def score_fn(resp, gt):
            return float(compute_score(
                solution_str=resp,
                ground_truth=str(gt),
            ))

        def extract_check(resp):
            return "\\boxed" in resp

        return score_fn, extract_check

    if scorer == "gsm8k_flexible":
        from verl.utils.reward_score.gsm8k import compute_score, extract_solution

        def score_fn(resp, gt):
            return float(compute_score(
                solution_str=resp,
                ground_truth=str(gt),
                method="flexible",
            ))

        def extract_check(resp):
            return extract_solution(resp, method="flexible") is not None

        return score_fn, extract_check

    raise ValueError(f"Unknown scorer: {scorer}")


def _score_outputs(dataset_key, output_path, n_samples, model_tag, tag):
    """Score generated responses using the dataset's configured scorer."""
    import os
    import json
    import pandas as pd
    import numpy as np

    cfg = DATASETS[dataset_key]
    score_fn, extract_check = _get_score_fn(cfg)
    print(f"[score] Using scorer '{cfg['scorer']}' for dataset '{dataset_key}'")

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
                score = score_fn(resp, gt)
            except Exception as e:
                print(f"[score] score_fn raised on problem {i} sample {j}: {e}")
                score = 0.0
            if score == 0.0 and not extract_check(resp):
                extract_failures += 1
            correctness[i, j] = int(score == 1.0)

        per_problem_rows.append(
            {
                "problem_idx": int(i),
                "ground_truth": str(gt),
                "n_samples": n_samples,
                "n_correct": int(correctness[i].sum()),
                "accuracy": float(correctness[i].mean()),
            }
        )

    metrics = {
        "dataset": dataset_key,
        "model": model_tag,
        "tag": tag,
        "scorer": cfg["scorer"],
        "num_problems": int(num_problems),
        "n_samples": int(n_samples),
        "extract_failures": int(extract_failures),
        "pass@1_mean": float(correctness.mean()),
    }
    for k in (1, 4, 8, 16):
        if k <= n_samples:
            vals = [_pass_at_k(n_samples, int(correctness[i].sum()), k) for i in range(num_problems)]
            metrics[f"pass@{k}"] = float(np.mean(vals))

    metrics_path = os.path.join(DATA_DIR, f"metrics_{dataset_key}_{model_tag}_{tag}.json")
    per_problem_path = os.path.join(DATA_DIR, f"per_problem_{dataset_key}_{model_tag}_{tag}.csv")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(per_problem_rows).to_csv(per_problem_path, index=False)

    print("\n=== Eval Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[score] metrics  -> {metrics_path}")
    print(f"[score] per-prob -> {per_problem_path}")

    return metrics


# ----------------------------- Modal entrypoint -----------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    timeout=4 * 3600,
)
def run_eval(
    dataset: str,
    model: str,
    n_samples: int,
    num_problems,
    max_response_length: int,
    max_prompt_length: int,
    subset: str,
    split: str,
    output_tag: str,
    use_lm_eval: bool,
):
    import os

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if dataset not in DATASETS:
        raise ValueError(f"Unknown --dataset={dataset!r}; choose from {list(DATASETS)}")
    if model not in MODEL_IDS:
        raise ValueError(f"Unknown --model={model!r}; choose from {list(MODEL_IDS)}")

    dataset_models = MODEL_IDS[model]
    if dataset not in dataset_models:
        raise ValueError(f"Model {model!r} is not supported or defined for dataset {dataset!r}")

    model_id = dataset_models[dataset]
    tag = output_tag or f"n{n_samples}_l{max_response_length}"

    if use_lm_eval:
        import lm_eval

        if dataset != "gsm8k":
            raise ValueError("--use-lm-eval currently only supports gsm8k")

        print(f"[lm-eval] Running official gsm8k evaluation for {model_id}")
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=f"pretrained={model_id},trust_remote_code=True,tensor_parallel_size=1,gpu_memory_utilization=0.9",
            tasks=["gsm8k"],
            batch_size="auto",
            device="cuda:0",
        )

        import json
        print("\n=== Raw lm-eval results['results'] ===")
        print(json.dumps(results["results"], indent=2))

        gsm8k_results = results["results"].get("gsm8k", {})

        accuracy_key = None
        for candidate in [
            "exact_match,flexible-extract",
            "exact_match,strict-match",
            "exact_match",
            "acc",
        ]:
            if candidate in gsm8k_results:
                accuracy_key = candidate
                break

        if accuracy_key is None:
            for k in gsm8k_results.keys():
                if "exact_match" in k or "acc" in k:
                    accuracy_key = k
                    break

        accuracy_val = (
            float(gsm8k_results[accuracy_key]) if accuracy_key and accuracy_key in gsm8k_results else 0.0
        )

        metrics = {
            "dataset": dataset,
            "model": model,
            "tag": tag,
            "scorer": "lm-eval-gsm8k",
            "accuracy": accuracy_val,
            "accuracy_metric_used": accuracy_key,
            "results_dict": results["results"],
        }
        print("\n=== lm-eval Summary ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        vol.commit()
        return metrics

    # 1. Prepare parquet
    data_path, n_problems = _prepare_dataset(
        dataset_key=dataset,
        subset=subset,
        split=split,
        num_problems=num_problems,
        tag=tag,
    )

    # 2. Run generation
    output_path = os.path.join(
        DATA_DIR, f"{dataset}_{model}_{tag}_outputs.parquet"
    )
    batch_size = min(n_problems, 24)
    _run_generation(
        data_path=data_path,
        output_path=output_path,
        model_id=model_id,
        n_samples=n_samples,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )

    # 3. Score
    metrics = _score_outputs(
        dataset_key=dataset,
        output_path=output_path,
        n_samples=n_samples,
        model_tag=model,
        tag=tag,
    )

    vol.commit()
    return metrics


@app.local_entrypoint()
def main(
    dataset: str = "aime",
    model: str = "qwen",
    n_samples: int = -1,
    num_problems: int = -1,
    max_response_length: int = -1,
    max_prompt_length: int = 1024,
    subset: str = "all",
    split: str = "",
    output_tag: str = "",
    use_lm_eval: bool = False,
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown --dataset={dataset!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset]

    n_samples_v = cfg["default_n_samples"] if n_samples < 0 else n_samples
    max_response_length_v = (
        cfg["default_response_length"] if max_response_length < 0 else max_response_length
    )
    split_v = split or cfg["default_split"]
    num_problems_v = None if num_problems is None or num_problems < 0 else int(num_problems)

    mode = "lm-eval" if use_lm_eval else "custom"
    print(
        f"[main] dataset={dataset} model={model} n_samples={n_samples_v} "
        f"max_response_length={max_response_length_v} subset={subset} split={split_v} "
        f"num_problems={num_problems_v} scorer={cfg['scorer']} mode={mode}"
    )

    metrics = run_eval.remote(
        dataset=dataset,
        model=model,
        n_samples=n_samples_v,
        num_problems=num_problems_v,
        max_response_length=max_response_length_v,
        max_prompt_length=max_prompt_length,
        subset=subset,
        split=split_v,
        output_tag=output_tag,
        use_lm_eval=use_lm_eval,
    )
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
