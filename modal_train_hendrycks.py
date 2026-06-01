#!/usr/bin/env python3
"""
Modal training entrypoint for Hendrycks MATH trivia augmentation experiment.

Usage:
    modal run modal_train_hendrycks.py --track a                    # full run
    modal run modal_train_hendrycks.py --track b                    # full run
    modal run modal_train_hendrycks.py --track a --total-steps 2    # smoke test
"""

import modal

app = modal.App("cs224r-trivia-train-hendrycks")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir("../cs224r-project-e3", "/root/verl", copy=True)
    .run_commands("pip install -e /root/verl", "pip install seaborn")
    .add_local_file("scripts/grpo_hendrycks_a100.sh", "/root/grpo_hendrycks_a100.sh")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/hendrycks_math"
CKPT_ROOT = "/data/ckpts/math"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SAVE_FREQ = 25
COMMIT_INTERVAL_S = 300  # background vol.commit() every 5 minutes

TRACK_CONFIG = {
    "a": {"parquet": "train_clean.parquet", "exp_name": "math-clean"},
    "b": {"parquet": "train_trivia_only.parquet", "exp_name": "math-trivia-only"},
}


# ---------------------------------------------------------------------------
# Volume commit safety
# ---------------------------------------------------------------------------
def _start_periodic_commit(volume, stop_event):
    import time
    import threading

    def _loop():
        while not stop_event.is_set():
            stop_event.wait(COMMIT_INTERVAL_S)
            if not stop_event.is_set():
                try:
                    volume.commit()
                    print(f"[commit] Periodic volume commit at {time.strftime('%H:%M:%S')}")
                except Exception as e:
                    print(f"[commit] Periodic commit failed: {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _install_sigterm_handler(volume):
    import signal
    import sys

    def _handler(signum, frame):
        print("[signal] SIGTERM received — committing volume...")
        try:
            volume.commit()
            print("[signal] Volume committed.")
        except Exception as e:
            print(f"[signal] Commit failed: {e}")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_train(track: str, total_steps: int, test_freq: int):
    import os
    import subprocess
    import threading

    if track not in TRACK_CONFIG:
        raise ValueError(f"Unknown --track={track!r}; choose 'a' or 'b'")

    config = TRACK_CONFIG[track]

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    train_parquet = os.path.join(DATA_DIR, config["parquet"])
    val_parquet = os.path.join(DATA_DIR, "test.parquet")
    exp_name = config["exp_name"]
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name)

    os.makedirs(ckpt_dir, exist_ok=True)

    for p in (train_parquet, val_parquet):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing parquet: {p}. Run: modal run scripts/data_upload_hendrycks.py"
            )

    env = os.environ.copy()
    env.update({
        "TRAIN_PARQUET": train_parquet,
        "VAL_PARQUET": val_parquet,
        "BASE_MODEL": BASE_MODEL,
        "CKPT_DIR": ckpt_dir,
        "EXPERIMENT_NAME": exp_name,
        "TOTAL_STEPS": str(total_steps),
        "SAVE_FREQ": str(SAVE_FREQ),
        "TEST_FREQ": str(test_freq),
        "WANDB_PROJECT": "cs224r-trivia-hendrycks",
    })

    subprocess.run(["sed", "-i", "s/\r//g", "/root/grpo_hendrycks_a100.sh"])

    print(f"[train] Track {'A' if track == 'a' else 'B'} | {exp_name} | "
          f"steps={total_steps} save_freq={SAVE_FREQ} test_freq={test_freq}")
    print(f"[train] Periodic commit every {COMMIT_INTERVAL_S}s")

    _install_sigterm_handler(vol)
    stop_event = threading.Event()
    _start_periodic_commit(vol, stop_event)

    try:
        subprocess.run(
            ["bash", "/root/grpo_hendrycks_a100.sh"],
            check=True, env=env, cwd="/root/verl",
        )
    finally:
        stop_event.set()
        vol.commit()
        print("[train] Final volume commit done.")

    return {"track": track, "exp_name": exp_name, "ckpt_dir": ckpt_dir}


@app.local_entrypoint()
def main(track: str = "a", total_steps: int = 400, test_freq: int = 25):
    print(f"[main] track={track} total_steps={total_steps}")
    result = run_train.remote(track=track, total_steps=total_steps, test_freq=test_freq)
    print(f"[main] done: {result}")
