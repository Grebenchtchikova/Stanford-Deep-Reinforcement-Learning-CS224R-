import pandas as pd
import glob
import os

files = sorted(glob.glob("*_outputs.parquet"))
if not files:
    print("No *_outputs.parquet files found in current directory.")
    exit()

for f in files:
    df = pd.read_parquet(f)
    name = f.split("_outputs")[0].replace("aime_", "")
    
    # responses is a list of 8 strings per row — explode it
    responses = df["responses"].explode()
    lengths = responses.str.len()
    
    print(f"{name}")
    print(f"  Problems: {len(df)}, Total responses: {len(responses)}")
    print(f"  Mean response length: {lengths.mean():.0f} chars")
    print(f"  Median: {lengths.median():.0f} chars")
    print(f"  Min: {lengths.min():.0f}, Max: {lengths.max():.0f}")
    print()
