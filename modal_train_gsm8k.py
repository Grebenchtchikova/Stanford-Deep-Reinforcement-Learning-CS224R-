#!/usr/bin/env python3
"""
Modal training entrypoint for the GSM8K trivia augmentation experiment.
 
Runs verl GRPO on a single H100 for either:
  - track a (clean)       : uses /data/gsm8k/train_clean.parquet
  - track b (trivia_only) : uses /data/gsm8k/train_trivia_only.parquet
 
Both tracks share:
  - base model      Qwen/Qwen3-1.7B
  - val parquet      /data/gsm8k/test.parquet
  - step budget      --total-steps (default 400)
 
Usage:
    modal run --detach modal_train_gsm8k.py --track a
    modal run --detach modal_train_gsm8k.py --track b
    modal run --detach modal_train_gsm8k.py --track a --total-steps 2   # smoke test
"""
 
import modal
 
app = modal.App("cs224r-trivia-train-gsm8k")
 
image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .pip_install("verl==0.2.0.post2")
    .add_local_file(
        "reward/gsm8k_custom.py",
        "/usr/local/lib/python3.10/dist-packages/verl/utils/reward_score/gsm8k_custom.py",
    )
    .add_local_file("scripts/grpo_gsm8k_a100.sh", "/root/grpo_gsm8k_a100.sh")
)
 
vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)
 
DATA_DIR = "/data/gsm8k"
CKPT_ROOT = "/data/ckpts/gsm8k"
BASE_MODEL = "Qwen/Qwen3-1.7B"
 
TRACK_CONFIG = {
    "a": {
        "parquet": "train_clean.parquet",
        "exp_name": "gsm8k-clean",
    },
    "b": {
        "parquet": "train_trivia_only.parquet",
        "exp_name": "gsm8k-trivia-only",
    },
}
 
 
@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_train(track: str, total_steps: int, save_freq: int, test_freq: int):
    import os
    import subprocess
 
    if track not in TRACK_CONFIG:
        raise ValueError(f"Unknown --track={track!r}; choose 'a' or 'b'")
 
    config = TRACK_CONFIG[track]
 
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
    train_parquet = os.path.join(DATA_DIR, config["parquet"])
    val_parquet = os.path.join(DATA_DIR, "test.parquet")
    exp_name = config["exp_name"]
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name)
 
    os.makedirs(ckpt_dir, exist_ok=True)
 
    # Verify data exists on volume
    for p in (train_parquet, val_parquet):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing parquet on Volume: {p}. "
                f"Run: modal run scripts/data_upload_gsm8k.py"
            )
 
    env = os.environ.copy()
    env.update({
        "TRAIN_PARQUET": train_parquet,
        "VAL_PARQUET": val_parquet,
        "BASE_MODEL": BASE_MODEL,
        "CKPT_DIR": ckpt_dir,
        "EXPERIMENT_NAME": exp_name,
        "TOTAL_STEPS": str(total_steps),
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": str(test_freq),
        "WANDB_PROJECT": "cs224r-trivia-gsm8k",
    })
 
    cmd = ["bash", "/root/grpo_gsm8k_a100.sh"]
 
    print(f"[modal_train] track={track} exp={exp_name} steps={total_steps}")
    print(f"[modal_train] train_parquet={train_parquet}")
    print(f"[modal_train] val_parquet={val_parquet}")
    print(f"[modal_train] ckpt_dir={ckpt_dir}")
 
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        vol.commit()
 
    return {"track": track, "experiment_name": exp_name, "ckpt_dir": ckpt_dir}
 
 
@app.local_entrypoint()
def main(
    track: str = "a",
    total_steps: int = 400,
    save_freq: int = 100,
    test_freq: int = 25,
):
    print(f"[main] track={track} total_steps={total_steps}")
    result = run_train.remote(
        track=track,
        total_steps=total_steps,
        save_freq=save_freq,
        test_freq=test_freq,
    )
    print(f"[main] done: {result}")
