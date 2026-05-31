"""
scripts/explore_dataset.py
SignVerse – Dataset exploration and statistics.

Usage
-----
    python scripts/explore_dataset.py --csv path/to/train.csv [OPTIONS]

Options
-------
    --csv          Path to train.csv                   (required)
    --output-dir   Directory for saved plots           (default: outputs/plots)
    --max-seq-len  Truncation passed to SignDataset    (default: 256)
    --top-k        Top-K tokens shown in freq plots    (default: 40)
    --dpi          Plot resolution                     (default: 150)
    --no-plots     Print stats only, skip rendering

Saved plots
-----------
    outputs/plots/seq_length_histogram.png
    outputs/plots/token_freq_base.png
    outputs/plots/token_freq_residuals.png
    outputs/plots/rvq_layer_stats.png
    outputs/plots/layer_vocab_utilization.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

# ── Make SignDataset importable regardless of working directory ───────────────
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datasets.sign_dataset import LAYER_COLUMNS, SignDataset

logger = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────────

PALETTE = {
    "base":       "#3266ad",
    "residuals":  ["#1d9e75", "#7f77dd", "#d85a30", "#d4537e", "#ba7517"],
    "neutral":    "#888780",
    "accent":     "#3c82c8",
    "grid":       "#e8e6df",
    "bg":         "#ffffff",
    "text":       "#2c2c2a",
    "text_light": "#5f5e5a",
}

LAYER_COLORS = [PALETTE["base"]] + PALETTE["residuals"]

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["neutral"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "xtick.color":       PALETTE["text_light"],
    "ytick.color":       PALETTE["text_light"],
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "medium",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PALETTE["bg"],
})

# ── Statistics ────────────────────────────────────────────────────────────────


def compute_sequence_stats(lengths: np.ndarray) -> dict:
    """Descriptive statistics for a 1-D array of sequence lengths."""
    percentiles = np.percentile(lengths, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count":  int(len(lengths)),
        "mean":   float(np.mean(lengths)),
        "std":    float(np.std(lengths)),
        "min":    int(np.min(lengths)),
        "p10":    int(percentiles[0]),
        "p25":    int(percentiles[1]),
        "median": int(percentiles[2]),
        "p75":    int(percentiles[3]),
        "p90":    int(percentiles[4]),
        "p95":    int(percentiles[5]),
        "p99":    int(percentiles[6]),
        "max":    int(np.max(lengths)),
    }


def compute_token_freq(tokens_2d: np.ndarray, col: int) -> Counter:
    """Return a Counter of token IDs for a single RVQ layer column."""
    return Counter(tokens_2d[:, col].tolist())


def compute_rvq_layer_stats(tokens_list: list[torch.Tensor]) -> list[dict]:
    """Per-layer descriptive statistics over all token ID values."""
    stats = []
    for col_idx, col_name in enumerate(LAYER_COLUMNS):
        ids = np.concatenate(
            [t[:, col_idx].numpy() for t in tokens_list]
        )
        freq   = Counter(ids.tolist())
        counts = np.array(list(freq.values()))
        stats.append({
            "layer":          col_name,
            "total_tokens":   int(len(ids)),
            "vocab_used":     int(len(freq)),
            "vocab_pct":      round(100 * len(freq) / 512, 1),
            "mean_id":        round(float(np.mean(ids)), 1),
            "std_id":         round(float(np.std(ids)), 1),
            "min_id":         int(np.min(ids)),
            "max_id":         int(np.max(ids)),
            "entropy_bits":   round(float(_entropy_bits(counts)), 3),
            "top1_id":        int(freq.most_common(1)[0][0]),
            "top1_freq_pct":  round(100 * freq.most_common(1)[0][1] / len(ids), 2),
        })
    return stats


def _entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a raw count array."""
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ── Printers ──────────────────────────────────────────────────────────────────


def print_sequence_stats(stats: dict) -> None:
    sep = "─" * 52
    print(f"\n{'Sequence length statistics':}")
    print(sep)
    rows = [
        ("Samples",   f"{stats['count']:,}"),
        ("Mean",      f"{stats['mean']:.1f} tokens"),
        ("Std dev",   f"{stats['std']:.1f} tokens"),
        ("Min",       f"{stats['min']} tokens"),
        ("p10",       f"{stats['p10']} tokens"),
        ("p25",       f"{stats['p25']} tokens"),
        ("Median",    f"{stats['median']} tokens"),
        ("p75",       f"{stats['p75']} tokens"),
        ("p90",       f"{stats['p90']} tokens"),
        ("p95",       f"{stats['p95']} tokens"),
        ("p99",       f"{stats['p99']} tokens"),
        ("Max",       f"{stats['max']} tokens"),
    ]
    for label, value in rows:
        print(f"  {label:<12}  {value}")
    print(sep)


def print_rvq_layer_stats(layer_stats: list[dict]) -> None:
    sep = "─" * 90
    header = (
        f"  {'Layer':<15} {'Total':>10} {'Vocab/512':>10} {'Mean ID':>8} "
        f"{'Std ID':>7} {'Entropy':>8} {'Top-1 %':>8}"
    )
    print(f"\n{'RVQ layer statistics':}")
    print(sep)
    print(header)
    print(sep)
    for s in layer_stats:
        print(
            f"  {s['layer']:<15} {s['total_tokens']:>10,} "
            f"  {s['vocab_used']:>3}/{512} ({s['vocab_pct']:>4.1f}%) "
            f"  {s['mean_id']:>7.1f}  {s['std_id']:>7.1f} "
            f"  {s['entropy_bits']:>7.3f}b  {s['top1_freq_pct']:>6.2f}%"
        )
    print(sep)


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_seq_length_histogram(
    lengths: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Sequence length distribution with percentile annotations."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Bin width = 8 frames
    bins = np.arange(0, lengths.max() + 8, 8)
    n, edges, patches = ax.hist(
        lengths, bins=bins,
        color=PALETTE["base"], alpha=0.85, linewidth=0,
    )

    # Colour the extreme-outlier tail differently
    p99 = np.percentile(lengths, 99)
    for patch, left in zip(patches, edges[:-1]):
        if left >= p99:
            patch.set_facecolor(PALETTE["residuals"][2])
            patch.set_alpha(0.7)

    # Percentile lines
    for pct, label, ls in [
        (50, "p50", "--"),
        (90, "p90", "-."),
        (99, "p99", ":"),
    ]:
        val = np.percentile(lengths, pct)
        ax.axvline(val, color=PALETTE["text"], linewidth=1.1, linestyle=ls, alpha=0.7)
        ax.text(
            val + 2, ax.get_ylim()[1] * 0.92,
            f"{label}={val:.0f}",
            fontsize=8, color=PALETTE["text_light"],
            va="top",
        )

    ax.set_xlabel("Sequence length (tokens = frames)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Sequence length distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Legend for tail colour
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=PALETTE["base"],             alpha=0.85, label="Normal range"),
        Patch(color=PALETTE["residuals"][2],     alpha=0.7,  label=f"Outliers (≥ p99 = {p99:.0f})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_token_freq_base(
    freq: Counter,
    output_path: Path,
    top_k: int = 40,
    dpi: int = 150,
) -> None:
    """Horizontal bar chart of top-K base_token IDs by frequency."""
    top = freq.most_common(top_k)
    ids    = [str(t[0]) for t in top]
    counts = [t[1]      for t in top]
    total  = sum(freq.values())

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.22)))
    y = np.arange(len(ids))

    bars = ax.barh(y, counts, height=0.65, color=PALETTE["base"], alpha=0.85)

    # Annotate bars with percentage
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_width() + total * 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%",
            va="center", fontsize=7, color=PALETTE["text_light"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences")
    ax.set_title(f"Top-{top_k} base_token IDs by frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_token_freq_residuals(
    all_freqs: list[Counter],
    output_path: Path,
    top_k: int = 40,
    dpi: int = 150,
) -> None:
    """5-panel grid: one top-K frequency bar chart per residual layer."""
    residual_names = LAYER_COLUMNS[1:]   # residual_1 … residual_5
    residual_freqs = all_freqs[1:]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    fig.suptitle(f"Top-{top_k} token IDs – residual layers 1–5", fontsize=11, y=1.01)

    for ax, freq, name, color in zip(
        axes, residual_freqs, residual_names, PALETTE["residuals"]
    ):
        top    = freq.most_common(top_k)
        ids    = [str(t[0]) for t in top]
        counts = [t[1]      for t in top]
        y = np.arange(len(ids))

        ax.barh(y, counts, height=0.65, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(ids, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Occurrences", fontsize=8)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
        )
        ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_rvq_layer_stats(
    layer_stats: list[dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """2×2 panel: mean ID, std ID, entropy, and top-1 frequency per layer."""
    metrics = [
        ("mean_id",       "Mean token ID",         "Mean ID value"),
        ("std_id",        "Std dev of token IDs",  "Std dev"),
        ("entropy_bits",  "Shannon entropy (bits)", "Bits"),
        ("top1_freq_pct", "Top-1 token dominance",  "% of total tokens"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    x      = np.arange(len(layer_stats))
    labels = [s["layer"].replace("_", "\n") for s in layer_stats]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [s[key] for s in layer_stats]
        bars   = ax.bar(x, values, color=LAYER_COLORS, alpha=0.85, width=0.55)

        # Value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=8, color=PALETTE["text_light"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values) * 1.18)

    fig.suptitle("RVQ layer statistics across all 6 codebook layers", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_layer_vocab_utilization(
    layer_stats: list[dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Horizontal stacked bar: used vs unused codebook slots per layer."""
    used   = [s["vocab_used"]       for s in layer_stats]
    unused = [512 - s["vocab_used"] for s in layer_stats]
    labels = [s["layer"]            for s in layer_stats]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(labels))

    ax.barh(y, used,   height=0.55, color=PALETTE["base"],             label="Used slots",   alpha=0.85)
    ax.barh(y, unused, height=0.55, color=PALETTE["grid"],             label="Unused slots", alpha=0.9,
            left=used)

    # Annotate used count and percentage
    for i, (u, s) in enumerate(zip(used, layer_stats)):
        ax.text(
            u / 2, i,
            f"{u}  ({s['vocab_pct']}%)",
            ha="center", va="center", fontsize=8.5,
            color="#ffffff", fontweight="medium",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 512)
    ax.set_xlabel("Codebook slots (total = 512)")
    ax.set_title("Codebook (vocab) utilisation per RVQ layer")
    ax.axvline(512, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved → %s", output_path)


# ── Orchestrator ──────────────────────────────────────────────────────────────


def explore(
    csv_path: Path,
    output_dir: Path,
    max_seq_len: int = 256,
    top_k: int = 40,
    dpi: int = 150,
    no_plots: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading dataset: {csv_path}")
    dataset = SignDataset(csv_path=csv_path, max_seq_len=max_seq_len)
    print(dataset.summary())

    tokens_list: list[torch.Tensor] = [s["tokens"] for s in dataset]

    # ── 2. Sequence length stats ──────────────────────────────────────────────
    lengths = np.array([t.shape[0] for t in tokens_list], dtype=np.int32)
    seq_stats = compute_sequence_stats(lengths)
    print_sequence_stats(seq_stats)

    # ── 3. All tokens as concatenated numpy array [N_total, 6] ───────────────
    print("\nConcatenating token tensors …", end=" ", flush=True)
    all_tokens_np: np.ndarray = np.concatenate(
        [t.numpy() for t in tokens_list], axis=0
    )  # [sum(T_i), 6]
    print(f"done  ({all_tokens_np.shape[0]:,} total frames × 6 layers)")

    # ── 4. Per-layer token frequency counters ─────────────────────────────────
    print("Computing token frequencies …", end=" ", flush=True)
    all_freqs: list[Counter] = [
        compute_token_freq(all_tokens_np, col) for col in range(6)
    ]
    print("done")

    # ── 5. RVQ layer stats ────────────────────────────────────────────────────
    layer_stats = compute_rvq_layer_stats(tokens_list)
    print_rvq_layer_stats(layer_stats)

    # ── 6. Gloss statistics ───────────────────────────────────────────────────
    gloss_lengths = np.array(
        [len(s["gloss"].split()) for s in dataset], dtype=np.int32
    )
    gloss_stats = compute_sequence_stats(gloss_lengths)
    sep = "─" * 52
    print(f"\n{'Gloss word-count statistics':}")
    print(sep)
    for label, value in [
        ("Samples",  f"{gloss_stats['count']:,}"),
        ("Mean",     f"{gloss_stats['mean']:.1f} signs"),
        ("Median",   f"{gloss_stats['median']} signs"),
        ("Max",      f"{gloss_stats['max']} signs"),
        ("Std dev",  f"{gloss_stats['std']:.1f} signs"),
    ]:
        print(f"  {label:<12}  {value}")
    print(sep)

    # ── 7. Avg tokens per gloss sign ──────────────────────────────────────────
    frames_per_sign = lengths / np.maximum(gloss_lengths, 1)
    print(
        f"\n  Average frames per gloss sign : {frames_per_sign.mean():.1f}  "
        f"(std={frames_per_sign.std():.1f}  "
        f"min={frames_per_sign.min():.1f}  "
        f"max={frames_per_sign.max():.1f})"
    )

    if no_plots:
        print("\n--no-plots set: skipping plot generation.")
        return

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print(f"\nSaving plots to {output_dir} …")

    plot_seq_length_histogram(
        lengths,
        output_path=output_dir / "seq_length_histogram.png",
        dpi=dpi,
    )

    plot_token_freq_base(
        all_freqs[0],
        output_path=output_dir / "token_freq_base.png",
        top_k=top_k,
        dpi=dpi,
    )

    plot_token_freq_residuals(
        all_freqs,
        output_path=output_dir / "token_freq_residuals.png",
        top_k=top_k,
        dpi=dpi,
    )

    plot_rvq_layer_stats(
        layer_stats,
        output_path=output_dir / "rvq_layer_stats.png",
        dpi=dpi,
    )

    plot_layer_vocab_utilization(
        layer_stats,
        output_path=output_dir / "layer_vocab_utilization.png",
        dpi=dpi,
    )

    print(f"\nAll plots saved to {output_dir}/")
    print("  seq_length_histogram.png")
    print("  token_freq_base.png")
    print("  token_freq_residuals.png")
    print("  rvq_layer_stats.png")
    print("  layer_vocab_utilization.png")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignVerse dataset explorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to train.csv",
    )
    parser.add_argument(
        "--output-dir", default="outputs/plots",
        help="Directory for saved PNG plots",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=256,
        help="Sequence truncation limit passed to SignDataset",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Number of top tokens to show in frequency plots",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Plot resolution (dots per inch)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Print statistics only, skip plot generation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    args = _parse_args(argv)

    explore(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        max_seq_len=args.max_seq_len,
        top_k=args.top_k,
        dpi=args.dpi,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
