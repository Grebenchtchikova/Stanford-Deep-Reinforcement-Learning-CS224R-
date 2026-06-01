#!/usr/bin/env python3
"""
Modal training entrypoint for Track E: Chung's original Track B replicated.

This is the 2M mixed dataset (M clean + M trivia copies) with partial e3
hyperparameters (negative gradients + loose clipping). This replicates
Chung's original setup exactly, to verify his results and compare against
Anna's controlled experiments.

Usage:
    modal run --detach modal_train_gsm8k_track_e.py
    modal run modal_train_gsm8k_track_e.py --total-steps 2   # smoke test
"""

import modal

app = modal.App("cs224r-track-e-train-gsm8k")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir("../cs224r-project-e3", "/root/verl", copy=True)
    .run_commands("pip install -e /root/verl", "pip install seaborn")
    .add_local_file(
        "reward/gsm8k_custom.py",
        "/root/verl/verl/utils/reward_score/gsm8k_custom.py",
    )
    .add_local_file("scripts/grpo_gsm8k_partial_e3.sh", "/root/grpo_gsm8k_partial_e3.sh")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/gsm8k"
CKPT_ROOT = "/data/ckpts/gsm8k"
BASE_MODEL = "Qwen/Qwen3-1.7B"

EXP_NAME = "gsm8k-track-e-2M-partial-e3"


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_train(total_steps: int, save_freq: int, test_freq: int):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_parquet = os.path.join(DATA_DIR, "train_mixed.parquet")
    val_parquet = os.path.join(DATA_DIR, "test.parquet")
    ckpt_dir = os.path.join(CKPT_ROOT, EXP_NAME)

    os.makedirs(ckpt_dir, exist_ok=True)

    # Verify data exists on volume
    for p in (train_parquet, val_parquet):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing parquet on Volume: {p}. "
                f"Run: modal run scripts/data_upload_gsm8k_mixed.py"
            )

    env = os.environ.copy()
    env.update({
        "TRAIN_PARQUET": train_parquet,
        "VAL_PARQUET": val_parquet,
        "BASE_MODEL": BASE_MODEL,
        "CKPT_DIR": ckpt_dir,
        "EXPERIMENT_NAME": EXP_NAME,
        "TOTAL_STEPS": str(total_steps),
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": str(test_freq),
        "WANDB_PROJECT": "cs224r-trivia-gsm8k",
    })

    # Fix Windows line endings
    subprocess.run(["sed", "-i", "s/\r//g", "/root/grpo_gsm8k_partial_e3.sh"])
    cmd = ["bash", "/root/grpo_gsm8k_partial_e3.sh"]

    print(f"[modal_train] Track E (Chung replica) steps={total_steps}")
    print(f"[modal_train] train_parquet={train_parquet} (2M mixed)")
    print(f"[modal_train] val_parquet={val_parquet}")
    print(f"[modal_train] ckpt_dir={ckpt_dir}")

    try:
        subprocess.run(cmd, check=True, env=env, cwd="/root/verl")
    finally:
        vol.commit()

    return {"experiment_name": EXP_NAME, "ckpt_dir": ckpt_dir}


@app.local_entrypoint()
def main(
    total_steps: int = 400,
    save_freq: int = 100,
    test_freq: int = 25,
):
    print(f"[main] Track E total_steps={total_steps}")
    result = run_train.remote(
        total_steps=total_steps,
        save_freq=save_freq,
        test_freq=test_freq,
    )
    print(f"[main] done: {result}")
