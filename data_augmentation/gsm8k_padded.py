#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GSM8K preprocessor for the RLAD noise-control experiment.
 
Modes:
    --mode clean      : standard GSM8K train set, M rows  -> Track A
    --mode mixed      : M originals + M fact-padded clones = 2M rows -> Track B (DEPRECATED: confounded by 2x data)
    --mode trivia_only: M rows, each with a random trivia fact prepended -> Track B (controlled)
 
The test split is always written clean (used for in-loop validation during GRPO
and final eval).
 
Output (under --local_dir):
    {local_dir}/train_clean.parquet          (mode=clean)  OR
    {local_dir}/train_mixed.parquet          (mode=mixed)  OR
    {local_dir}/train_trivia_only.parquet    (mode=trivia_only)
    {local_dir}/test.parquet                 (always)
 
Padded rows prepend one random trivia fact (each <=15 tokens, no numerals)
followed by a space to the question text. Ground truth is unchanged.
"""
 
import argparse
import os
import random
import re
 
import datasets
 
 
# 20 trivia facts: each short, no numerals (to avoid contaminating either the
# strict scorer at train time or the flexible scorer at eval time).
TRIVIA_FACTS = [
    "Water freezes at zero degrees.",
    "The sky often appears blue.",
    "Honey never spoils.",
    "Bananas are technically berries.",
    "Octopuses have three hearts.",
    "Sharks existed before trees.",
    "Sound travels faster in water.",
    "A group of crows is called a murder.",
    "Penguins propose with pebbles.",
    "Sloths can hold their breath long.",
    "Cats cannot taste sweetness.",
    "Wombats produce cube-shaped droppings.",
    "Lightning is hotter than the sun's surface.",
    "Glass is technically a slow liquid.",
    "Butterflies taste with their feet.",
    "Owls cannot move their eyeballs.",
    "Bees can recognize human faces.",
    "Sea otters hold hands while sleeping.",
    "Some turtles breathe through their cloaca.",
    "Dolphins have names for each other.",
]
 
 
def extract_solution(solution_str: str) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None, f"GSM8K answer regex failed on: {solution_str!r}"
    final_solution = solution.group(0).split("#### ")[1].replace(",", "")
    return final_solution
 
 
INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer after \"####\"."
 
 
def build_row(question_text: str, gt: str, split: str, idx: int, raw_answer: str, raw_question: str):
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": f"{question_text} {INSTRUCTION_FOLLOWING}"}],
        "ability": "math",
        "level": "unknown",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": raw_answer,
            "question": raw_question,
        },
    }
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clean", "mixed", "trivia_only"], required=True)
    parser.add_argument("--local_dir", default="./data_gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
 
    os.makedirs(args.local_dir, exist_ok=True)
    rng = random.Random(args.seed)
 
    print(f"[gsm8k_padded] Loading openai/gsm8k (config=main)...")
    ds = datasets.load_dataset("openai/gsm8k", "main")
    train_raw = ds["train"]
    test_raw = ds["test"]
    print(f"[gsm8k_padded] train={len(train_raw)} test={len(test_raw)}")
 
    # ---- Build train rows ----
    train_rows = []
 
    if args.mode == "trivia_only":
        # Controlled experiment: M rows, every problem gets trivia prepended.
        # Same data budget as Track A (clean), only difference is noise.
        for idx, ex in enumerate(train_raw):
            q = ex["question"]
            a = ex["answer"]
            gt = extract_solution(a)
            fact = rng.choice(TRIVIA_FACTS)
            padded_q = f"{fact} {q}"
            train_rows.append(build_row(padded_q, gt, "train", idx, a, padded_q))
        print(f"[gsm8k_padded] trivia_only: {len(train_rows)} rows (each with trivia prepended)")
 
    else:
        # clean or mixed: start with all clean rows
        for idx, ex in enumerate(train_raw):
            q = ex["question"]
            a = ex["answer"]
            gt = extract_solution(a)
            train_rows.append(build_row(q, gt, "train", idx, a, q))
 
        if args.mode == "mixed":
            M = len(train_rows)
            for idx, ex in enumerate(train_raw):
                q = ex["question"]
                a = ex["answer"]
                gt = extract_solution(a)
                fact = rng.choice(TRIVIA_FACTS)
                padded_q = f"{fact} {q}"
                train_rows.append(build_row(padded_q, gt, "train", M + idx, a, padded_q))
            print(f"[gsm8k_padded] mixed: kept {M} originals + {M} padded clones = {len(train_rows)}")
 
    # ---- Build test rows (always clean) ----
    test_rows = []
    for idx, ex in enumerate(test_raw):
        q = ex["question"]
        a = ex["answer"]
        gt = extract_solution(a)
        test_rows.append(build_row(q, gt, "test", idx, a, q))
 
    # ---- Write parquets ----
    train_names = {
        "clean": "train_clean.parquet",
        "mixed": "train_mixed.parquet",
        "trivia_only": "train_trivia_only.parquet",
    }
    train_name = train_names[args.mode]
    train_path = os.path.join(args.local_dir, train_name)
    test_path = os.path.join(args.local_dir, "test.parquet")
 
    train_ds = datasets.Dataset.from_list(train_rows)
    test_ds = datasets.Dataset.from_list(test_rows)
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
 
    print(f"[gsm8k_padded] wrote {len(train_rows)} train rows -> {train_path}")
    print(f"[gsm8k_padded] wrote {len(test_rows)} test rows  -> {test_path}")
 
 
if __name__ == "__main__":
    main()
