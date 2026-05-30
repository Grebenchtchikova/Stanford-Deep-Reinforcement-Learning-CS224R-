#!/usr/bin/env python3
"""
Post-evaluation analysis for the trivia augmentation experiment.
 
Computes and compares across tracks:
  1. pass@1 accuracy (from eval metrics JSONs)
  2. Response length distribution
  3. Cosine similarity of solution traces (diversity measure)
  4. Per-token entropy (requires logits — runs a separate inference pass)
 
Usage:
    # Download eval outputs from Modal volume first:
    modal volume get cs224r-trivia-vol /data/eval/ ./results/
 
    # Then run analysis:
    python analyze_results.py --results-dir ./results/
 
    # Or run on Modal directly:
    modal run analyze_results.py
"""
 
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
 
 
# ----------------------------- Metrics loading -----------------------------
 
def load_metrics(results_dir, dataset="gsm8k"):
    """Load all metrics JSONs for a dataset and return as a dict keyed by model."""
    metrics = {}
    for fname in os.listdir(results_dir):
        if fname.startswith(f"metrics_{dataset}_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                m = json.load(f)
                metrics[m["model"]] = m
    return metrics
 
 
def load_responses(results_dir, dataset="gsm8k"):
    """Load all output parquets for a dataset, keyed by model tag."""
    responses = {}
    for fname in os.listdir(results_dir):
        if fname.startswith(f"{dataset}_") and fname.endswith("_outputs.parquet"):
            parts = fname.replace(f"{dataset}_", "").replace("_outputs.parquet", "")
            model_tag = parts.rsplit("_", 1)[0] if "_" in parts else parts
            df = pd.read_parquet(os.path.join(results_dir, fname))
            responses[model_tag] = df
    return responses
 
 
# ----------------------------- Response length -----------------------------
 
def compute_response_lengths(responses_df):
    """Compute token-approximate lengths (whitespace split) for all responses."""
    lengths = []
    for _, row in responses_df.iterrows():
        for resp in row["responses"]:
            lengths.append(len(resp.split()))
    return np.array(lengths)
 
 
def compare_response_lengths(responses_dict):
    """Print and return length statistics per model."""
    stats = {}
    for model, df in responses_dict.items():
        lengths = compute_response_lengths(df)
        stats[model] = {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "p90": float(np.percentile(lengths, 90)),
        }
        print(f"  {model}: mean={stats[model]['mean']:.0f}, "
              f"median={stats[model]['median']:.0f}, "
              f"std={stats[model]['std']:.0f}, "
              f"p90={stats[model]['p90']:.0f}")
    return stats
 
 
# ----------------------------- Cosine similarity -----------------------------
 
def _tfidf_vector(text, vocab):
    """Simple TF vector for a response using a shared vocabulary."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    vec = np.zeros(len(vocab))
    for token, idx in vocab.items():
        if token in counts:
            vec[idx] = counts[token]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
 
 
def compute_solution_diversity(responses_df, max_problems=None):
    """
    Compute mean pairwise cosine similarity of solution traces per problem.
    Lower similarity = more diverse solutions = less policy collapse.
    """
    problems = responses_df.head(max_problems) if max_problems else responses_df
    per_problem_sim = []
 
    for _, row in problems.iterrows():
        resps = row["responses"]
        if len(resps) < 2:
            continue
 
        # Build vocabulary from all responses for this problem
        all_tokens = set()
        for r in resps:
            all_tokens.update(r.lower().split())
        vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
 
        # Compute pairwise cosine similarity
        vecs = [_tfidf_vector(r, vocab) for r in resps]
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = np.dot(vecs[i], vecs[j])
                sims.append(sim)
 
        per_problem_sim.append(np.mean(sims))
 
    return np.array(per_problem_sim)
 
 
def compare_solution_diversity(responses_dict, max_problems=200):
    """Print diversity statistics per model."""
    stats = {}
    for model, df in responses_dict.items():
        sims = compute_solution_diversity(df, max_problems=max_problems)
        if len(sims) == 0:
            print(f"  {model}: no multi-sample problems found")
            continue
        stats[model] = {
            "mean_cosine_sim": float(np.mean(sims)),
            "std_cosine_sim": float(np.std(sims)),
        }
        print(f"  {model}: mean_cosine_sim={stats[model]['mean_cosine_sim']:.4f} "
              f"(lower = more diverse)")
    return stats
 
 
# ----------------------------- Text entropy proxy -----------------------------
 
def compute_text_entropy(responses_df):
    """
    Compute token-level Shannon entropy of each response as a proxy for
    per-token model entropy. This measures vocabulary diversity in the output,
    not the model's predictive uncertainty — for true per-token entropy,
    logits are needed (see note below).
    """
    entropies = []
    for _, row in responses_df.iterrows():
        for resp in row["responses"]:
            tokens = resp.lower().split()
            if len(tokens) == 0:
                entropies.append(0.0)
                continue
            counts = Counter(tokens)
            total = len(tokens)
            probs = np.array([c / total for c in counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(entropy)
    return np.array(entropies)
 
 
def compare_text_entropy(responses_dict):
    """Print text entropy statistics per model."""
    stats = {}
    for model, df in responses_dict.items():
        ents = compute_text_entropy(df)
        stats[model] = {
            "mean_entropy": float(np.mean(ents)),
            "std_entropy": float(np.std(ents)),
        }
        print(f"  {model}: mean_text_entropy={stats[model]['mean_entropy']:.3f}")
    return stats
 
 
# ----------------------------- Plotting -----------------------------
 
def plot_comparison(metrics_dict, length_stats, diversity_stats, entropy_stats, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    models = list(metrics_dict.keys())
    colors = {"track_a": "#2563eb", "track_b": "#dc2626", "qwen": "#6b7280", "e3": "#059669"}
 
    # 1. pass@1 bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    pass1 = [metrics_dict[m].get("pass@1_mean", metrics_dict[m].get("pass@1", 0)) for m in models]
    bars = ax.bar(models, pass1, color=[colors.get(m, "#8b5cf6") for m in models])
    ax.set_ylabel("pass@1")
    ax.set_title("GSM8K pass@1 by Track")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, pass1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pass_at_1.png"), dpi=150)
    print(f"  saved pass_at_1.png")
 
    # 2. Response length comparison
    if length_stats:
        fig, ax = plt.subplots(figsize=(8, 5))
        means = [length_stats[m]["mean"] for m in models if m in length_stats]
        lbls = [m for m in models if m in length_stats]
        bars = ax.bar(lbls, means, color=[colors.get(m, "#8b5cf6") for m in lbls])
        ax.set_ylabel("Mean response length (whitespace tokens)")
        ax.set_title("Response Length by Track")
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "response_length.png"), dpi=150)
        print(f"  saved response_length.png")
 
    # 3. Solution diversity
    if diversity_stats:
        fig, ax = plt.subplots(figsize=(8, 5))
        sims = [diversity_stats[m]["mean_cosine_sim"] for m in models if m in diversity_stats]
        lbls = [m for m in models if m in diversity_stats]
        bars = ax.bar(lbls, sims, color=[colors.get(m, "#8b5cf6") for m in lbls])
        ax.set_ylabel("Mean pairwise cosine similarity")
        ax.set_title("Solution Trace Similarity (lower = more diverse)")
        for bar, v in zip(bars, sims):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "solution_diversity.png"), dpi=150)
        print(f"  saved solution_diversity.png")
 
    plt.close("all")
 
 
# ----------------------------- Main -----------------------------
 
def run_analysis(results_dir, dataset="gsm8k", output_dir="./figures"):
    print(f"\n{'='*60}")
    print(f"  CS224R Trivia Augmentation — {dataset.upper()} Analysis")
    print(f"{'='*60}")
 
    # Load metrics
    print("\n--- pass@1 Accuracy ---")
    metrics = load_metrics(results_dir, dataset)
    if not metrics:
        print(f"  No metrics found in {results_dir} for dataset={dataset}")
        return
    for model, m in metrics.items():
        p1 = m.get("pass@1_mean", m.get("pass@1", "N/A"))
        print(f"  {model}: pass@1 = {p1}")
 
    # Load responses
    print("\n--- Response Lengths ---")
    responses = load_responses(results_dir, dataset)
    length_stats = compare_response_lengths(responses) if responses else {}
 
    # Solution diversity
    print("\n--- Solution Diversity (cosine similarity) ---")
    diversity_stats = compare_solution_diversity(responses) if responses else {}
 
    # Text entropy proxy
    print("\n--- Text Entropy (vocabulary diversity proxy) ---")
    entropy_stats = compare_text_entropy(responses) if responses else {}
    print("  NOTE: This is a text-level proxy. For true per-token model entropy,")
    print("  logits must be extracted during generation (not yet implemented).")
 
    # Plots
    print(f"\n--- Generating plots -> {output_dir}/ ---")
    plot_comparison(metrics, length_stats, diversity_stats, entropy_stats, output_dir)
 
    # Summary JSON
    summary = {
        "dataset": dataset,
        "metrics": metrics,
        "response_lengths": length_stats,
        "solution_diversity": diversity_stats,
        "text_entropy": entropy_stats,
    }
    summary_path = os.path.join(output_dir, f"{dataset}_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--output-dir", default="./figures")
    args = parser.parse_args()
    run_analysis(args.results_dir, args.dataset, args.output_dir)
