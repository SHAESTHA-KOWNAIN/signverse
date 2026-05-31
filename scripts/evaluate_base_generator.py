"""
scripts/evaluate_base_generator.py
SignVerse – Diagnostic evaluator for BaseTokenGenerator.

Measures generation quality and diagnoses mode-collapse / repetition
in the trained BaseTokenGenerator.  Results are written to
``outputs/base_generator_diagnostics.json`` and printed to stdout.

Metrics collected
-----------------
Validation set (teacher-forcing):
    * cross-entropy loss
    * token accuracy   (non-PAD positions only)
    * perplexity       (exp(loss))

Generation diversity (greedy, 100 random validation sentences):
    * unique_sequences          – distinct generated token sequences
    * pct_identical_outputs     – % of outputs identical to the most common one
    * avg_sequence_entropy      – mean Shannon entropy (bits) per sequence
    * avg_token_diversity       – mean fraction of unique tokens per sequence
    * avg_pct_repeated_tokens   – mean % of token positions that are a repetition
                                  of the immediately preceding token
    * top_generated_tokens      – 20 most frequent generated token IDs

Ground-truth comparison (greedy vs target):
    * avg_token_overlap_pct     – mean % of generated tokens that appear in
                                  the target set for the same sentence
    * avg_length_ratio          – mean (generated_len / target_len)

Qualitative examples:
    * 20 randomly sampled sentence / target / generated triples

Usage
-----
    # Full evaluation (requires train.csv and best.pt):
    python scripts/evaluate_base_generator.py             \\
        --csv   data/train.csv                            \\
        --ckpt  checkpoints/base_token_generator/best.pt

    # Adjust sample counts and output path:
    python scripts/evaluate_base_generator.py             \\
        --csv          data/train.csv                     \\
        --ckpt         checkpoints/base_token_generator/best.pt \\
        --n_gen        200                                \\
        --n_examples   30                                 \\
        --max_len      128                                \\
        --out          outputs/my_diagnostics.json

    # Self-contained smoke test (no CSV / checkpoint needed):
    python scripts/evaluate_base_generator.py smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

# ── Project imports ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.sign_dataset import SignDataset, collate_sign_batch
from models.base_token_generator import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    BaseTokenGenerator,
)
from models.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT:       str = "checkpoints/base_token_generator/best.pt"
DEFAULT_N_GEN:      int = 100     # sentences used for generation-diversity analysis
DEFAULT_N_EXAMPLES: int = 20      # qualitative examples printed / saved
DEFAULT_MAX_LEN:    int = 128     # max tokens to generate per sentence
DEFAULT_VAL_FRAC:   float = 0.20
DEFAULT_SEED:       int = 42
DEFAULT_BATCH:      int = 16
DEFAULT_WORKERS:    int = 4
DEFAULT_OUT:        str = "outputs/base_generator_diagnostics.json"


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────


def load_models(
    ckpt_path:  str,
    device:     torch.device,
) -> tuple[TextEncoder, BaseTokenGenerator, dict]:
    """Load TextEncoder and BaseTokenGenerator from a training checkpoint.

    Parameters
    ----------
    ckpt_path:
        Path to ``best.pt`` produced by ``train_base_token_generator.py``.
    device:
        Target device.

    Returns
    -------
    encoder, generator, config
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train first:  python training/train_base_token_generator.py --csv data/train.csv"
        )

    ckpt   = torch.load(path, map_location="cpu", weights_only=True)
    config = ckpt["config"]

    logger.info(
        "Checkpoint: %s  epoch=%d  best_val_loss=%.6f",
        path, ckpt["epoch"], ckpt["best_val_loss"],
    )

    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=True,
        device=str(device),
    )
    encoder.eval()

    generator = BaseTokenGenerator(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=0.0,
        max_seq_len=config["model_max_seq_len"],
        text_dim=encoder.hidden_dim,
    ).to(device)
    generator.load_state_dict(ckpt["model_state"])
    generator.eval()

    logger.info("Models ready.  %s", generator)
    return encoder, generator, config


# ──────────────────────────────────────────────────────────────────────────────
# Validation-set metric computation  (teacher-forcing)
# ──────────────────────────────────────────────────────────────────────────────


def _build_decoder_tensors(
    tokens:  torch.Tensor,   # [B, T_max, 6]
    lengths: torch.Tensor,   # [B]
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift base tokens into (decoder_input, decoder_target) pair.

    Returns
    -------
    dec_input  : LongTensor [B, T_max]  – [BOS, t_0, …, t_{T-2}]
    dec_target : LongTensor [B, T_max]  – [t_0,  t_1, …, t_{T-1}]
    """
    base = tokens[:, :, 0].to(device)     # [B, T_max]
    B, T = base.shape
    bos  = base.new_full((B, 1), BOS_TOKEN_ID)
    dec_input  = torch.cat([bos, base[:, :-1]], dim=1)   # [B, T]
    dec_target = base.clone()                             # [B, T]

    # Enforce PAD beyond true lengths
    pad_mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.to(device).unsqueeze(1)
    dec_input  = dec_input.masked_fill(pad_mask, PAD_TOKEN_ID)
    dec_target = dec_target.masked_fill(pad_mask, PAD_TOKEN_ID)
    dec_input[:, 0] = BOS_TOKEN_ID    # position 0 is always BOS
    return dec_input, dec_target


@torch.no_grad()
def compute_val_metrics(
    encoder:   TextEncoder,
    generator: BaseTokenGenerator,
    loader:    DataLoader,
    device:    torch.device,
) -> dict[str, float]:
    """Compute loss, token accuracy, and perplexity on a validation DataLoader.

    Parameters
    ----------
    encoder:
        Frozen TextEncoder in eval mode.
    generator:
        BaseTokenGenerator in eval mode.
    loader:
        Validation DataLoader (collate_sign_batch).
    device:
        Compute device.

    Returns
    -------
    dict with keys: ``loss``, ``accuracy``, ``perplexity``.
    """
    try:
        from tqdm import tqdm
        wrapped = tqdm(loader, desc="Val metrics", leave=False, dynamic_ncols=True)
    except ImportError:
        wrapped = loader

    total_loss = 0.0
    total_corr = 0
    total_real = 0
    n_batches  = 0

    for batch in wrapped:
        text_emb, text_mask = encoder.encode(batch["sentences"])
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        dec_input, dec_target = _build_decoder_tensors(
            batch["tokens"], batch["lengths"], device
        )

        logits = generator(text_emb, text_mask, dec_input)   # [B, T, VOCAB_SIZE]

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            dec_target.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
            reduction="mean",
        )

        preds     = logits.reshape(-1, VOCAB_SIZE).argmax(dim=-1)
        flat_tgt  = dec_target.reshape(-1)
        real_mask = flat_tgt != PAD_TOKEN_ID
        total_corr += ((preds == flat_tgt) & real_mask).sum().item()
        total_real += real_mask.sum().item()
        total_loss += loss.item()
        n_batches  += 1

    avg_loss   = total_loss / max(1, n_batches)
    accuracy   = (total_corr / max(1, total_real)) * 100.0
    perplexity = math.exp(min(avg_loss, 300.0))

    return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity}


# ──────────────────────────────────────────────────────────────────────────────
# Per-sequence diagnostic metrics
# ──────────────────────────────────────────────────────────────────────────────


def _sequence_entropy(tokens: list[int]) -> float:
    """Shannon entropy (bits) of the token distribution in one sequence.

    A perfectly uniform sequence over V distinct tokens has entropy log2(V).
    A sequence that always repeats the same token has entropy 0.

    Parameters
    ----------
    tokens:
        Non-empty list of integer token IDs (PAD already stripped).

    Returns
    -------
    Entropy in bits, or 0.0 for an empty or length-1 sequence.
    """
    if len(tokens) <= 1:
        return 0.0
    counts = Counter(tokens)
    n      = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _pct_repeated(tokens: list[int]) -> float:
    """Fraction (%) of positions that are identical to the preceding token.

    Position 0 is never counted as repeated.  Returns 0.0 for sequences
    shorter than 2 tokens.
    """
    if len(tokens) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return (reps / (len(tokens) - 1)) * 100.0


def _token_diversity(tokens: list[int]) -> float:
    """Fraction of positions occupied by a unique token (type-token ratio).

    Returns 0.0 for empty sequences.
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _overlap_pct(generated: list[int], target: list[int]) -> float:
    """% of generated tokens whose ID appears anywhere in the target sequence."""
    if not generated:
        return 0.0
    target_set = set(target)
    hits = sum(1 for t in generated if t in target_set)
    return (hits / len(generated)) * 100.0


def _strip_pad(tokens: list[int]) -> list[int]:
    """Remove trailing PAD_TOKEN_ID values."""
    out = tokens[:]
    while out and out[-1] == PAD_TOKEN_ID:
        out.pop()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Generation-diversity analysis
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def analyse_generation(
    encoder:   TextEncoder,
    generator: BaseTokenGenerator,
    samples:   list[dict],           # list of SignDataset items (raw dicts)
    max_len:   int,
    device:    torch.device,
    n_examples: int,
    seed:       int,
) -> dict:
    """Generate tokens for *samples* and compute diversity / comparison metrics.

    Each sample is processed individually (batch size 1) so that the generated
    length is not influenced by padding from other sequences.

    Parameters
    ----------
    encoder:
        Frozen TextEncoder.
    generator:
        BaseTokenGenerator in eval mode.
    samples:
        List of raw SignDataset sample dicts (``sentence``, ``tokens``).
    max_len:
        Maximum tokens to generate per sentence.
    device:
        Compute device.
    n_examples:
        Number of random qualitative examples to include in the report.
    seed:
        RNG seed for reproducible example selection.

    Returns
    -------
    dict containing diversity metrics, comparison metrics, and example triples.
    """
    rng = random.Random(seed)

    generated_seqs: list[list[int]] = []
    target_seqs:    list[list[int]] = []
    sentences:      list[str]       = []

    try:
        from tqdm import tqdm
        iter_samples = tqdm(samples, desc="Generating", leave=False, dynamic_ncols=True)
    except ImportError:
        iter_samples = samples

    for sample in iter_samples:
        sentence = sample["sentence"]
        # Ground-truth base tokens (strip PAD)
        target = _strip_pad(sample["tokens"][:, 0].tolist())

        text_emb, text_mask = encoder.encode([sentence])
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        gen_raw = generator.generate(
            text_embeddings=text_emb,
            text_padding_mask=text_mask,
            max_len=max_len,
            strategy="greedy",
        )  # [1, max_len]
        generated = _strip_pad(gen_raw[0].tolist())

        generated_seqs.append(generated)
        target_seqs.append(target)
        sentences.append(sentence)

    n = len(generated_seqs)
    logger.info("Generated %d sequences.", n)

    # ── Uniqueness ────────────────────────────────────────────────────────
    frozen    = [tuple(s) for s in generated_seqs]
    counts    = Counter(frozen)
    n_unique  = len(counts)
    most_common_seq, most_common_count = counts.most_common(1)[0]
    pct_identical = (most_common_count / max(1, n)) * 100.0

    # ── Per-sequence diagnostics ──────────────────────────────────────────
    entropies:   list[float] = [_sequence_entropy(s)  for s in generated_seqs]
    diversities: list[float] = [_token_diversity(s)   for s in generated_seqs]
    repeats:     list[float] = [_pct_repeated(s)      for s in generated_seqs]

    avg_entropy   = sum(entropies)   / max(1, n)
    avg_diversity = sum(diversities) / max(1, n)
    avg_repeated  = sum(repeats)     / max(1, n)

    # ── Token frequency ───────────────────────────────────────────────────
    all_gen_tokens: list[int] = [t for s in generated_seqs for t in s]
    top_tokens = [
        {"token_id": int(tid), "count": int(cnt)}
        for tid, cnt in Counter(all_gen_tokens).most_common(20)
    ]

    # ── Ground-truth comparison ───────────────────────────────────────────
    overlaps:       list[float] = [_overlap_pct(g, t) for g, t in zip(generated_seqs, target_seqs)]
    length_ratios:  list[float] = [
        len(g) / max(1, len(t)) for g, t in zip(generated_seqs, target_seqs)
    ]
    avg_overlap      = sum(overlaps)      / max(1, n)
    avg_length_ratio = sum(length_ratios) / max(1, n)

    # ── Per-target diversity (how varied are the ground-truth sequences?) ─
    target_entropies  = [_sequence_entropy(t) for t in target_seqs]
    avg_target_entropy = sum(target_entropies) / max(1, n)

    # ── Qualitative examples ──────────────────────────────────────────────
    indices  = list(range(n))
    rng.shuffle(indices)
    ex_idxs  = indices[:n_examples]

    examples = []
    for idx in sorted(ex_idxs):
        gen = generated_seqs[idx]
        tgt = target_seqs[idx]
        examples.append({
            "sentence":        sentences[idx],
            "target_length":   len(tgt),
            "generated_length": len(gen),
            "target_tokens":   tgt[:32],   # cap display at 32 tokens
            "generated_tokens": gen[:32],
            "sequence_entropy":  round(_sequence_entropy(gen),  4),
            "token_diversity":   round(_token_diversity(gen),    4),
            "pct_repeated":      round(_pct_repeated(gen),       2),
            "overlap_pct":       round(_overlap_pct(gen, tgt),   2),
            "length_ratio":      round(len(gen) / max(1, len(tgt)), 4),
        })

    return {
        # Uniqueness
        "n_samples":             n,
        "n_unique_sequences":    n_unique,
        "pct_unique_sequences":  round((n_unique / max(1, n)) * 100.0, 2),
        "pct_identical_outputs": round(pct_identical, 2),
        "most_common_sequence":  list(most_common_seq)[:32],
        # Diversity
        "avg_sequence_entropy":    round(avg_entropy,   4),
        "avg_token_diversity":     round(avg_diversity, 4),
        "avg_pct_repeated_tokens": round(avg_repeated,  2),
        # Ground-truth comparison
        "avg_token_overlap_pct":   round(avg_overlap,       2),
        "avg_length_ratio":        round(avg_length_ratio,  4),
        "avg_target_entropy":      round(avg_target_entropy, 4),
        # Token frequency
        "top_generated_tokens":    top_tokens,
        # Qualitative
        "examples":                examples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

_WIDE = "═" * 68
_THIN = "─" * 68


def _print_report(val_metrics: dict, gen_metrics: dict) -> None:
    """Print the full diagnostic report to stdout."""
    print()
    print(_WIDE)
    print("  SignVerse – BaseTokenGenerator Diagnostics")
    print(_WIDE)

    # ── Validation metrics ────────────────────────────────────────────────
    print("\n  VALIDATION METRICS  (teacher-forcing on held-out split)")
    print(_THIN)
    print(f"  Loss        : {val_metrics['loss']:.6f}")
    print(f"  Perplexity  : {val_metrics['perplexity']:.2f}")
    print(f"  Accuracy    : {val_metrics['accuracy']:.2f}%")

    # ── Generation diversity ──────────────────────────────────────────────
    print(f"\n  GENERATION DIVERSITY  (greedy, n={gen_metrics['n_samples']})")
    print(_THIN)
    print(f"  Unique sequences      : {gen_metrics['n_unique_sequences']} / {gen_metrics['n_samples']}"
          f"  ({gen_metrics['pct_unique_sequences']:.1f}%)")
    print(f"  Identical to mode     : {gen_metrics['pct_identical_outputs']:.1f}%")
    print(f"  Avg sequence entropy  : {gen_metrics['avg_sequence_entropy']:.4f} bits")
    print(f"  Avg token diversity   : {gen_metrics['avg_token_diversity']:.4f}  (type-token ratio)")
    print(f"  Avg repeated tokens   : {gen_metrics['avg_pct_repeated_tokens']:.1f}%")

    # ── Ground-truth comparison ───────────────────────────────────────────
    print(f"\n  GROUND-TRUTH COMPARISON")
    print(_THIN)
    print(f"  Avg token overlap     : {gen_metrics['avg_token_overlap_pct']:.1f}%")
    print(f"  Avg length ratio      : {gen_metrics['avg_length_ratio']:.4f}  (gen / target)")
    print(f"  Avg target entropy    : {gen_metrics['avg_target_entropy']:.4f} bits")

    # ── Top generated tokens ──────────────────────────────────────────────
    print(f"\n  TOP-20 GENERATED TOKEN IDs")
    print(_THIN)
    top = gen_metrics["top_generated_tokens"]
    total_gen = sum(t["count"] for t in top)
    rows = [
        f"  {t['token_id']:>4}  ×{t['count']:<6}  ({t['count']/max(1,total_gen)*100:.1f}%)"
        for t in top
    ]
    # Two-column layout
    half = math.ceil(len(rows) / 2)
    for i in range(half):
        left  = rows[i]
        right = rows[i + half] if (i + half) < len(rows) else ""
        print(f"{left:<36}{right}")

    # ── Qualitative examples ──────────────────────────────────────────────
    print(f"\n  QUALITATIVE EXAMPLES  ({len(gen_metrics['examples'])} random samples)")
    print(_THIN)
    for i, ex in enumerate(gen_metrics["examples"], start=1):
        print(f"\n  [{i:>2}] Sentence   : {ex['sentence']!r}")
        print(f"       Target len : {ex['target_length']}   Generated len: {ex['generated_length']}"
              f"   Ratio: {ex['length_ratio']:.2f}")
        print(f"       Entropy    : {ex['sequence_entropy']:.3f} bits   "
              f"Diversity: {ex['token_diversity']:.3f}   "
              f"Repeated: {ex['pct_repeated']:.1f}%   "
              f"Overlap: {ex['overlap_pct']:.1f}%")
        tgt_disp = str(ex["target_tokens"])
        gen_disp = str(ex["generated_tokens"])
        print(f"       Target     : {tgt_disp}")
        print(f"       Generated  : {gen_disp}")

    # ── Diagnosis summary ─────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS")
    print(_THIN)
    _print_diagnosis(val_metrics, gen_metrics)
    print()
    print(_WIDE)
    print()


def _print_diagnosis(val_metrics: dict, gen_metrics: dict) -> None:
    """Print a plain-English diagnostic summary of observed failure modes."""
    issues: list[str] = []

    if gen_metrics["pct_identical_outputs"] > 50.0:
        issues.append(
            f"  ⚠ MODE COLLAPSE: {gen_metrics['pct_identical_outputs']:.1f}% of generated "
            f"sequences are identical to the most common output.\n"
            f"    → Try: lower learning rate, increase dropout, reduce label smoothing,\n"
            f"           add top-k/nucleus sampling at inference, train longer."
        )
    if gen_metrics["avg_pct_repeated_tokens"] > 30.0:
        issues.append(
            f"  ⚠ REPETITION: {gen_metrics['avg_pct_repeated_tokens']:.1f}% of token positions "
            f"repeat the previous token.\n"
            f"    → Try: repetition penalty during generation, lower temperature,\n"
            f"           check for degenerate loss plateau (loss ≈ log(512) ≈ 6.24)."
        )
    if gen_metrics["avg_sequence_entropy"] < 1.0:
        issues.append(
            f"  ⚠ LOW ENTROPY: avg sequence entropy is {gen_metrics['avg_sequence_entropy']:.3f} bits.\n"
            f"    → Model may have collapsed to a near-constant output distribution.\n"
            f"    → Reference: uniform over 512 tokens → entropy ≈ 9.0 bits."
        )
    if gen_metrics["avg_token_overlap_pct"] < 10.0:
        issues.append(
            f"  ⚠ LOW OVERLAP: generated tokens overlap ground-truth by only "
            f"{gen_metrics['avg_token_overlap_pct']:.1f}%.\n"
            f"    → Model has not learned the target distribution yet."
        )
    if val_metrics["perplexity"] > 400.0:
        issues.append(
            f"  ⚠ HIGH PERPLEXITY: {val_metrics['perplexity']:.1f} "
            f"(random baseline over 512 tokens ≈ 512).\n"
            f"    → Model is near random; may need more epochs or a lower LR."
        )
    if gen_metrics["avg_length_ratio"] < 0.3 or gen_metrics["avg_length_ratio"] > 3.0:
        issues.append(
            f"  ⚠ LENGTH MISMATCH: avg generated/target length ratio = "
            f"{gen_metrics['avg_length_ratio']:.2f}.\n"
            f"    → Consider conditioning generation on LengthPredictor output."
        )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No critical failure modes detected.")
        print(f"    Perplexity={val_metrics['perplexity']:.1f}  "
              f"Accuracy={val_metrics['accuracy']:.1f}%  "
              f"Unique={gen_metrics['pct_unique_sequences']:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(
    csv_path:   str,
    ckpt_path:  str  = DEFAULT_CKPT,
    n_gen:      int  = DEFAULT_N_GEN,
    n_examples: int  = DEFAULT_N_EXAMPLES,
    max_len:    int  = DEFAULT_MAX_LEN,
    val_frac:   float = DEFAULT_VAL_FRAC,
    seed:       int  = DEFAULT_SEED,
    batch_size: int  = DEFAULT_BATCH,
    workers:    int  = DEFAULT_WORKERS,
    out_path:   str  = DEFAULT_OUT,
    device_str: str  = "auto",
) -> dict:
    """Run the full evaluation pipeline and save the diagnostics report.

    Parameters
    ----------
    csv_path:
        Path to ``train.csv``.
    ckpt_path:
        Trained checkpoint path.
    n_gen:
        Number of validation sentences to use for generation-diversity analysis.
    n_examples:
        Number of qualitative examples to include in the report.
    max_len:
        Maximum tokens to generate per sentence.
    val_frac:
        Fraction of the dataset held out as validation (must match training).
    seed:
        RNG seed (must match the seed used during training for the same split).
    batch_size:
        Batch size for teacher-forcing validation pass.
    workers:
        DataLoader worker count.
    out_path:
        Path to write the JSON report.
    device_str:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns
    -------
    dict with keys ``val_metrics`` and ``generation_metrics``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)

    torch.manual_seed(seed)

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", csv_path)
    dataset = SignDataset(
        csv_path=csv_path,
        max_seq_len=256,
        skip_corrupt_gloss=True,
    )
    logger.info(dataset.summary())

    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    gen_rng = torch.Generator().manual_seed(seed)
    _, val_sub = random_split(dataset, [n_train, n_val], generator=gen_rng)
    logger.info("Validation split: %d samples", len(val_sub))

    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sign_batch,
        pin_memory=device.type == "cuda",
    )

    # ── Models ────────────────────────────────────────────────────────────
    encoder, generator, config = load_models(ckpt_path, device)

    # ── Validation metrics ────────────────────────────────────────────────
    logger.info("Computing validation metrics (teacher-forcing) …")
    val_metrics = compute_val_metrics(encoder, generator, val_loader, device)
    logger.info(
        "Val: loss=%.4f  acc=%.2f%%  ppl=%.2f",
        val_metrics["loss"], val_metrics["accuracy"], val_metrics["perplexity"],
    )

    # ── Generation analysis ───────────────────────────────────────────────
    # Draw a random subset of the validation split for generation
    rng_local    = random.Random(seed)
    val_indices  = list(range(len(val_sub)))
    rng_local.shuffle(val_indices)
    gen_indices  = val_indices[:n_gen]

    # Retrieve raw samples (SignDataset items) from the Subset
    gen_samples: list[dict] = [val_sub.dataset[val_sub.indices[i]] for i in gen_indices]

    logger.info("Running greedy generation on %d sentences …", len(gen_samples))
    gen_metrics = analyse_generation(
        encoder=encoder,
        generator=generator,
        samples=gen_samples,
        max_len=max_len,
        device=device,
        n_examples=n_examples,
        seed=seed,
    )

    # ── Report ────────────────────────────────────────────────────────────
    _print_report(val_metrics, gen_metrics)

    report = {
        "checkpoint":         ckpt_path,
        "epoch":              int(torch.load(ckpt_path, map_location="cpu", weights_only=True)["epoch"]),
        "config":             config,
        "val_metrics":        val_metrics,
        "generation_metrics": gen_metrics,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Diagnostics saved → %s", out_file)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Validate every metric function and the reporting pipeline with synthetic data.

    No checkpoint, CSV, or GPU is required.

    Assertions
    ----------
    * All per-sequence metric helpers produce values in their expected ranges.
    * ``analyse_generation`` returns the correct structure and metric bounds.
    * ``compute_val_metrics`` accumulates loss/accuracy correctly on synthetic batches.
    * ``_print_report`` runs to completion without exceptions.
    * ``_print_diagnosis`` correctly identifies injected failure modes.
    """
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("=" * 60)
    logger.info("SMOKE TEST: evaluate_base_generator.py")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ── Metric helpers ────────────────────────────────────────────────────
    logger.info("── Metric helpers ──────────────────────────────────────────")

    # _sequence_entropy
    assert _sequence_entropy([]) == 0.0
    assert _sequence_entropy([1]) == 0.0
    assert abs(_sequence_entropy([1, 2, 3, 4]) - 2.0) < 1e-6, \
        "Uniform 4-token seq should have entropy 2.0 bits"
    assert _sequence_entropy([7, 7, 7, 7]) == 0.0, \
        "Constant sequence has 0 entropy"
    logger.info("  _sequence_entropy: PASSED")

    # _pct_repeated
    assert _pct_repeated([])           == 0.0
    assert _pct_repeated([1])          == 0.0
    assert _pct_repeated([1, 1, 1, 1]) == 100.0
    assert _pct_repeated([1, 2, 3, 4]) == 0.0
    assert abs(_pct_repeated([1, 1, 2, 2]) - 66.666) < 0.1, \
        "[1,1,2,2] → 2/3 repeated"
    logger.info("  _pct_repeated: PASSED")

    # _token_diversity
    assert _token_diversity([])           == 0.0
    assert _token_diversity([5, 5, 5])    == pytest_approx(1 / 3) if False \
        else abs(_token_diversity([5, 5, 5]) - (1 / 3)) < 1e-6
    assert _token_diversity([1, 2, 3])    == 1.0
    logger.info("  _token_diversity: PASSED")

    # _overlap_pct
    assert _overlap_pct([], [1, 2]) == 0.0
    assert _overlap_pct([1, 2], [1, 2]) == 100.0
    assert _overlap_pct([3, 4], [1, 2]) == 0.0
    assert abs(_overlap_pct([1, 3], [1, 2]) - 50.0) < 1e-6
    logger.info("  _overlap_pct: PASSED")

    # _strip_pad
    assert _strip_pad([1, 2, PAD_TOKEN_ID])         == [1, 2]
    assert _strip_pad([PAD_TOKEN_ID])               == []
    assert _strip_pad([1, 2, 3])                    == [1, 2, 3]
    assert _strip_pad([])                           == []
    logger.info("  _strip_pad: PASSED")

    # ── _build_decoder_tensors ────────────────────────────────────────────
    logger.info("── _build_decoder_tensors ──────────────────────────────────")
    B, T_max = 3, 8
    tokens_6  = torch.randint(0, 512, (B, T_max, 6))
    lengths   = torch.tensor([8, 5, 3])
    for i, rlen in enumerate(lengths.tolist()):
        tokens_6[i, rlen:, :] = PAD_TOKEN_ID
    di, dt = _build_decoder_tensors(tokens_6, lengths, device)
    assert di.shape == (B, T_max)
    assert (di[:, 0] == BOS_TOKEN_ID).all(), "Position 0 must be BOS"
    for i, rlen in enumerate(lengths.tolist()):
        if rlen < T_max:
            assert (dt[i, rlen:] == PAD_TOKEN_ID).all(), \
                f"Target positions beyond length must be PAD for sample {i}"
    logger.info("  _build_decoder_tensors: PASSED  shape=%s", tuple(di.shape))

    # ── analyse_generation with stub models ───────────────────────────────
    logger.info("── analyse_generation ──────────────────────────────────────")
    D_txt = 768

    class _StubEncoder:
        hidden_dim = D_txt
        def encode(self, sents: list[str]):
            T = 8
            emb = torch.randn(len(sents), T, self.hidden_dim, device=device)
            msk = torch.zeros(len(sents), T, dtype=torch.bool, device=device)
            return emb, msk

    tiny_gen = BaseTokenGenerator(
        hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128,
        dropout=0.0, max_seq_len=256, text_dim=D_txt,
    ).to(device)
    tiny_gen.eval()

    # Build 12 synthetic "samples" matching SignDataset item structure
    T_mot = 15
    syn_samples = []
    for k in range(12):
        toks = torch.randint(0, 512, (T_mot, 6))
        syn_samples.append({"sentence": f"Sentence {k}", "tokens": toks})

    result = analyse_generation(
        encoder=_StubEncoder(),          # type: ignore[arg-type]
        generator=tiny_gen,
        samples=syn_samples,
        max_len=20,
        device=device,
        n_examples=5,
        seed=42,
    )

    assert result["n_samples"]            == 12
    assert 0 <= result["n_unique_sequences"] <= 12
    assert 0.0 <= result["pct_identical_outputs"] <= 100.0
    assert result["avg_sequence_entropy"] >= 0.0
    assert 0.0 <= result["avg_token_diversity"]     <= 1.0
    assert 0.0 <= result["avg_pct_repeated_tokens"] <= 100.0
    assert 0.0 <= result["avg_token_overlap_pct"]   <= 100.0
    assert result["avg_length_ratio"]     > 0.0
    assert len(result["examples"])        == 5
    assert len(result["top_generated_tokens"]) <= 20
    for ex in result["examples"]:
        assert "sentence"         in ex
        assert "target_tokens"    in ex
        assert "generated_tokens" in ex
        assert "overlap_pct"      in ex
    logger.info("  analyse_generation: PASSED  n_unique=%d  entropy=%.4f",
                result["n_unique_sequences"], result["avg_sequence_entropy"])

    # ── compute_val_metrics with synthetic batches ────────────────────────
    logger.info("── compute_val_metrics ─────────────────────────────────────")

    # Manually construct two batches to validate accumulation
    class _TwoItemLoader:
        """Fake DataLoader that yields two pre-built batches."""
        def __init__(self, batches): self._batches = batches
        def __iter__(self): return iter(self._batches)

    B2, T2 = 2, 6
    batch_a = {
        "sentences": ["Hello", "World"],
        "tokens":    torch.randint(0, 512, (B2, T2, 6)),
        "lengths":   torch.tensor([T2, T2]),
    }
    batch_b = {
        "sentences": ["Foo", "Bar"],
        "tokens":    torch.randint(0, 512, (B2, T2, 6)),
        "lengths":   torch.tensor([T2, T2]),
    }

    enc_stub  = _StubEncoder()
    fake_loader = _TwoItemLoader([batch_a, batch_b])

    val_out = compute_val_metrics(enc_stub, tiny_gen, fake_loader, device)  # type: ignore
    assert "loss"       in val_out
    assert "accuracy"   in val_out
    assert "perplexity" in val_out
    assert val_out["loss"]       > 0.0
    assert 0.0 <= val_out["accuracy"] <= 100.0
    assert val_out["perplexity"] > 1.0
    assert not math.isnan(val_out["loss"])
    assert not math.isnan(val_out["perplexity"])
    logger.info("  compute_val_metrics: PASSED  loss=%.4f  acc=%.2f%%  ppl=%.2f",
                val_out["loss"], val_out["accuracy"], val_out["perplexity"])

    # ── _print_report (no exceptions) ─────────────────────────────────────
    logger.info("── _print_report ───────────────────────────────────────────")
    _print_report(val_out, result)
    logger.info("  _print_report: PASSED")

    # ── _print_diagnosis detects injected failure modes ───────────────────
    logger.info("── _print_diagnosis failure-mode detection ─────────────────")
    import io, contextlib

    bad_val = {"loss": 6.25, "accuracy": 0.5, "perplexity": 520.0}
    bad_gen = {
        "pct_identical_outputs":   95.0,
        "avg_pct_repeated_tokens": 80.0,
        "avg_sequence_entropy":    0.05,
        "avg_token_overlap_pct":    2.0,
        "avg_length_ratio":         0.1,
        "pct_unique_sequences":     5.0,
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        _print_diagnosis(bad_val, bad_gen)
    output = captured.getvalue()
    assert "MODE COLLAPSE" in output,   "Should detect mode collapse"
    assert "REPETITION"    in output,   "Should detect repetition"
    assert "LOW ENTROPY"   in output,   "Should detect low entropy"
    assert "LOW OVERLAP"   in output,   "Should detect low overlap"
    assert "PERPLEXITY"    in output,   "Should detect high perplexity"
    assert "LENGTH MISMATCH" in output, "Should detect length mismatch"
    logger.info("  All 6 failure modes detected: PASSED")

    good_val = {"loss": 2.5, "accuracy": 45.0, "perplexity": 12.0}
    good_gen = {
        "pct_identical_outputs":   5.0,
        "avg_pct_repeated_tokens": 10.0,
        "avg_sequence_entropy":    5.0,
        "avg_token_overlap_pct":  55.0,
        "avg_length_ratio":        0.95,
        "pct_unique_sequences":   95.0,
    }
    captured2 = io.StringIO()
    with contextlib.redirect_stdout(captured2):
        _print_diagnosis(good_val, good_gen)
    assert "✓" in captured2.getvalue(), "Healthy model should get a pass tick"
    logger.info("  Healthy model pass: PASSED")

    # ── JSON serialisability ──────────────────────────────────────────────
    logger.info("── JSON serialisability ────────────────────────────────────")
    report = {"val_metrics": val_out, "generation_metrics": result}
    serialised = json.dumps(report, indent=2)
    roundtripped = json.loads(serialised)
    assert roundtripped["val_metrics"]["loss"] == val_out["loss"]
    logger.info("  JSON round-trip: PASSED")

    # ── Report save ───────────────────────────────────────────────────────
    logger.info("── report save ─────────────────────────────────────────────")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "outputs" / "diag.json"
        out_file.parent.mkdir(parents=True)
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        assert out_file.exists() and out_file.stat().st_size > 0
        loaded = json.loads(out_file.read_text())
        assert "val_metrics" in loaded
    logger.info("  Report save/load: PASSED")

    # ── load_models file-not-found guard ──────────────────────────────────
    logger.info("── load_models error handling ──────────────────────────────")
    try:
        load_models("/nonexistent/best.pt", device)
        assert False, "Should have raised"
    except FileNotFoundError as exc:
        logger.info("  FileNotFoundError: PASSED  (%s)", str(exc)[:60])

    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose SignVerse BaseTokenGenerator generation quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")
    sub.add_parser("smoke_test", help="Self-contained smoke test (no data needed)")

    p.add_argument("--csv",        required=False, default=None,
                   help="Path to train.csv (required unless smoke_test)")
    p.add_argument("--ckpt",       default=DEFAULT_CKPT,       dest="ckpt_path")
    p.add_argument("--n_gen",      default=DEFAULT_N_GEN,       type=int,
                   help="Sentences to use for generation-diversity analysis")
    p.add_argument("--n_examples", default=DEFAULT_N_EXAMPLES,  type=int,
                   help="Qualitative examples to print / save")
    p.add_argument("--max_len",    default=DEFAULT_MAX_LEN,     type=int,
                   help="Max tokens to generate per sentence")
    p.add_argument("--val_frac",   default=DEFAULT_VAL_FRAC,    type=float)
    p.add_argument("--seed",       default=DEFAULT_SEED,        type=int)
    p.add_argument("--batch",      default=DEFAULT_BATCH,       type=int)
    p.add_argument("--workers",    default=DEFAULT_WORKERS,     type=int)
    p.add_argument("--out",        default=DEFAULT_OUT,         dest="out_path",
                   help="Output JSON path")
    p.add_argument("--device",     default="auto")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode == "smoke_test":
        _smoke_test()
        return

    if not args.csv:
        parser.error("--csv is required for evaluation (use smoke_test for testing without data)")

    evaluate(
        csv_path=args.csv,
        ckpt_path=args.ckpt_path,
        n_gen=args.n_gen,
        n_examples=args.n_examples,
        max_len=args.max_len,
        val_frac=args.val_frac,
        seed=args.seed,
        batch_size=args.batch,
        workers=args.workers,
        out_path=args.out_path,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()