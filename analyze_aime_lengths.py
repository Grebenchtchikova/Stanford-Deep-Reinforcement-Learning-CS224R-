import pandas as pd
import glob
import os

files = sorted(glob.glob("*_outputs.parquet"))
if not files:
    print("No *_outputs.parquet files found in current directory.")
    print(f"Current directory: {os.getcwd()}")
    print("Make sure you ran: modal volume get cs224r-trivia-vol eval/aime/<filename> .")
else:
    for f in files:
        df = pd.read_parquet(f)
        print(f"\n{f}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        # Try common column names for the response text
        for col in ["response", "output", "completion", "generated_text"]:
            if col in df.columns:
                lengths = df[col].str.len()
                print(f"  Column '{col}': mean={lengths.mean():.0f} chars, median={lengths.median():.0f}")
                break
        else:
            # Just show all string columns and their lengths
            for col in df.columns:
                if df[col].dtype == object:
                    lengths = df[col].str.len()
                    print(f"  Column '{col}': mean={lengths.mean():.0f} chars, median={lengths.median():.0f}")
