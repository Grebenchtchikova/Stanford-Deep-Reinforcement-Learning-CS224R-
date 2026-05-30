#!/usr/bin/env python3
"""
Generate clean and trivia-only Hendrycks MATH training parquets and upload
them to a Modal Volume.

Note: hendrycks_padded.py imports from verl for boxed answer extraction,
so this script needs verl installed (unlike the GSM8K data upload).

Usage:
    modal run scripts/data_upload_hendrycks.py
    modal run scripts/data_upload_hendrycks.py --seed 123
"""

import modal

app = modal.App("cs224r-data-upload-hendrycks")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .run_commands("pip install --no-deps verl==0.2.0.post2")
    .add_local_file("data_augmentation/hendrycks_padded.py", "/root/hendrycks_padded.py")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/hendrycks_math"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=30 * 60,
)
def generate_data(seed: int):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.makedirs(DATA_DIR, exist_ok=True)

    for mode in ("clean", "trivia_only"):
        cmd = [
            "python3", "/root/hendrycks_padded.py",
            "--mode", mode,
            "--local_dir", DATA_DIR,
            "--seed", str(seed),
        ]
        print(f"[data_upload] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Verify outputs
    print(f"\n[data_upload] contents of {DATA_DIR}:")
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        size = os.path.getsize(full)
        print(f"  {name}  ({size:,} bytes)")

    vol.commit()


@app.local_entrypoint()
def main(seed: int = 42):
    generate_data.remote(seed=seed)
    print("[main] data upload complete")
