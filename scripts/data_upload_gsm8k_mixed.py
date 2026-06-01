#!/usr/bin/env python3
"""
Upload the 2M mixed GSM8K parquet (M clean + M trivia copies) to Modal volume.
This replicates Chung's original Track B data.

Usage:
    modal run scripts/data_upload_gsm8k_mixed.py
"""

import modal

app = modal.App("gsm8k-data-upload-mixed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets")
    .add_local_file("data_augmentation/gsm8k_padded.py", "/root/gsm8k_padded.py")
)

vol = modal.Volume.from_name("cs224r-trivia-vol", create_if_missing=True)

DATA_DIR = "/data/gsm8k"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=30 * 60,
)
def generate_data(seed: int):
    import os
    import subprocess

    os.makedirs(DATA_DIR, exist_ok=True)

    cmd = [
        "python3", "/root/gsm8k_padded.py",
        "--mode", "mixed",
        "--local_dir", DATA_DIR,
        "--seed", str(seed),
    ]
    print(f"[data_upload] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Verify
    print(f"\n[data_upload] contents of {DATA_DIR}:")
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        size = os.path.getsize(full)
        print(f"  {name}  ({size:,} bytes)")

    vol.commit()


@app.local_entrypoint()
def main(seed: int = 42):
    generate_data.remote(seed=seed)
    print("[main] mixed data upload complete")
