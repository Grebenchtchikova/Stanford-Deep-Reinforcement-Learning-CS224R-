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
Hendrycks MATH preprocessor for the RLAD noise-control experiment.

Modes:
    --mode clean      : standard MATH train set, M rows  -> Track A
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

Source dataset: DigitalLearningGmbH/MATH-lighteval (mirror of the original
lighteval/MATH, which is no longer hosted on HuggingFace).
"""

import argparse
import os
import random

import datasets

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


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


DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str: str) -> str:
    boxed = last_boxed_only_string(solution_str)
    assert boxed is not None, f"MATH answer extraction failed (no \\boxed{{}}): {solution_str!r}"
    return remove_boxed(boxed)


def build_row(question_text: str, gt: str, split: str, idx: int, raw_answer: str, raw_question: str):
    return {
        "data_source": DATA_SOURCE,
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
    parser.add_argument("--local_dir", default="./data_hendrycks_math")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[hendrycks_padded] Loading {DATA_SOURCE}...")
    ds = datasets.load_dataset(DATA_SOURCE, trust_remote_code=True)
    train_raw = ds["train"]
    test_raw = ds["test"]
    print(f"[hendrycks_padded] train={len(train_raw)} test={len(test_raw)}")

    # ---- Build train rows ----
    train_rows = []

    if args.mode == "trivia_only":
        # Controlled experiment: M rows, every problem gets trivia prepended.
        # Same data budget as Track A (clean), only difference is noise.
        for idx, ex in enumerate(train_raw):
            q = ex["problem"]
            a = ex["solution"]
            gt = extract_solution(a)
            fact = rng.choice(TRIVIA_FACTS)
            padded_q = f"{fact} {q}"
            train_rows.append(build_row(padded_q, gt, "train", idx, a, padded_q))
        print(f"[hendrycks_padded] trivia_only: {len(train_rows)} rows (each with trivia prepended)")

    else:
        # clean or mixed: start with all clean rows
        for idx, ex in enumerate(train_raw):
            q = ex["problem"]
            a = ex["solution"]
            gt = extract_solution(a)
            train_rows.append(build_row(q, gt, "train", idx, a, q))

        if args.mode == "mixed":
            M = len(train_rows)
            for idx, ex in enumerate(train_raw):
                q = ex["problem"]
                a = ex["solution"]
                gt = extract_solution(a)
                fact = rng.choice(TRIVIA_FACTS)
                padded_q = f"{fact} {q}"
                train_rows.append(build_row(padded_q, gt, "train", M + idx, a, padded_q))
            print(f"[hendrycks_padded] mixed: kept {M} originals + {M} padded clones = {len(train_rows)}")

    # ---- Build test rows (always clean) ----
    test_rows = []
    for idx, ex in enumerate(test_raw):
        q = ex["problem"]
        a = ex["solution"]
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

    print(f"[hendrycks_padded] wrote {len(train_rows)} train rows -> {train_path}")
    print(f"[hendrycks_padded] wrote {len(test_rows)} test rows  -> {test_path}")


if __name__ == "__main__":
    main()
