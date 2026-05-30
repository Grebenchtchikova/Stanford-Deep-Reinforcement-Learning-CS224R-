#!/usr/bin/env python3
"""
Modal helper to convert verl FSDP checkpoints into HF model directories for
post-training evaluation with modal_eval_general.py.

Locates the latest `global_step_*/actor` directory under the checkpoint path
and writes the merged HF model to <experiment_name>_hf/.

Usage:
    modal run modal_convert_ckpt.py --track a --dataset gsm8k
    modal run modal_convert_ckpt.py --track b --dataset gsm8k
    modal run modal_convert_ckpt.py --track a --dataset math
    modal run modal_convert_ckpt.py --track b --dataset math
    modal run modal_convert_ckpt.py --track a --dataset gsm8k --step 400
"""

import modal

app = modal.App("cs224r-trivia-convert")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .run_commands("pip install --no-deps verl==0.2.0.post2", "fire")
    .add_local_file("convert_fsdp_to_hf.py", "/root/convert_fsdp_to_hf.py")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3-1.7B"

TRACK_CONFIG = {
    "gsm8k": {
        "ckpt_root": "/data/ckpts/gsm8k",
        "a": "gsm8k-clean",
        "b": "gsm8k-trivia-only",
    },
    "math": {
        "ckpt_root": "/data/ckpts/math",
        "a": "math-clean",
        "b": "math-trivia-only",
    },
}


def _find_latest_step_dir(exp_dir: str) -> str:
    import os
    import re

    candidates = []
    for name in os.listdir(exp_dir):
        m = re.match(r"global_step_(\d+)$", name)
        if m:
            candidates.append((int(m.group(1)), name))
    if not candidates:
        raise FileNotFoundError(f"No global_step_* dirs in {exp_dir}")
    candidates.sort()
    return os.path.join(exp_dir, candidates[-1][1])


def _detect_world_size(actor_dir: str) -> int:
    import os
    import re

    sizes = set()
    for name in os.listdir(actor_dir):
        m = re.match(r"model_world_size_(\d+)_rank_\d+\.pt$", name)
        if m:
            sizes.add(int(m.group(1)))
    if not sizes:
        raise FileNotFoundError(
            f"No model_world_size_*_rank_*.pt shards in {actor_dir}"
        )
    if len(sizes) > 1:
        raise RuntimeError(f"Multiple world sizes detected in {actor_dir}: {sizes}")
    return sizes.pop()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def run_convert(track: str, dataset: str, step: int):
    import os
    import subprocess

    if dataset not in TRACK_CONFIG:
        raise ValueError(f"Unknown --dataset={dataset!r}; choose from {list(TRACK_CONFIG)}")

    cfg = TRACK_CONFIG[dataset]
    if track not in ("a", "b"):
        raise ValueError(f"Unknown --track={track!r}; choose 'a' or 'b'")

    os.environ["HF_HOME"] = "/data/hf_cache"

    exp_name = cfg[track]
    ckpt_root = cfg["ckpt_root"]
    exp_dir = os.path.join(ckpt_root, exp_name)

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"No checkpoint dir for track {track}: {exp_dir}")

    if step > 0:
        step_dir = os.path.join(exp_dir, f"global_step_{step}")
        if not os.path.isdir(step_dir):
            raise FileNotFoundError(f"Requested step dir missing: {step_dir}")
    else:
        step_dir = _find_latest_step_dir(exp_dir)
    print(f"[convert] using step dir: {step_dir}")

    actor_dir = os.path.join(step_dir, "actor")
    if not os.path.isdir(actor_dir):
        raise FileNotFoundError(f"Expected actor dir at {actor_dir}")

    world_size = _detect_world_size(actor_dir)
    print(f"[convert] detected world_size={world_size}")

    output_path = os.path.join(ckpt_root, f"{exp_name}_hf")
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python3",
        "/root/convert_fsdp_to_hf.py",
        "--fsdp_checkpoint_path", actor_dir,
        "--huggingface_model_path", BASE_MODEL,
        "--output_path", output_path,
        "--world_size", str(world_size),
    ]
    print(f"[convert] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"[convert] HF model written to {output_path}")
    return {"track": track, "dataset": dataset, "step_dir": step_dir, "hf_path": output_path}


@app.local_entrypoint()
def main(track: str = "a", dataset: str = "gsm8k", step: int = -1):
    """--step=-1 means latest available global_step_* dir."""
    result = run_convert.remote(track=track, dataset=dataset, step=step)
    print(f"[main] done: {result}")
