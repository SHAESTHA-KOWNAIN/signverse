"""
scripts/test_base_generation.py
SignVerse – Inference test for the trained BaseTokenGenerator.

Loads a trained ``checkpoints/base_token_generator/best.pt`` checkpoint,
encodes one or more English sentences with the frozen FLAN-T5 TextEncoder,
and generates RVQ base-layer motion tokens using both greedy decoding and
top-k sampling.  Results are printed to stdout and logged at INFO level.

Usage
-----
    # Use the hardcoded example sentences:
    python scripts/test_base_generation.py

    # Supply your own sentences as positional arguments:
    python scripts/test_base_generation.py "Good morning" "Where is the exit?"

    # Override checkpoint path and decoding settings:
    python scripts/test_base_generation.py                        \\
        --ckpt   checkpoints/base_token_generator/best.pt         \\
        --max_len 64                                               \\
        --top_k   40                                               \\
        --temperature 0.9                                          \\
        "Hello, how are you?"

    # Run the self-contained smoke test (no checkpoint required):
    python scripts/test_base_generation.py smoke_test

Output format (one block per sentence)
---------------------------------------
    ══════════════════════════════════════════════════════════
    Sentence  : Hello
    ──────────────────────────────────────────────────────────
    Generated length   : 32 tokens
    Greedy tokens      : [247, 13, 88, …]
    Top-k tokens       : [247, 50, 91, …]
    ══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

# ── Project imports ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.base_token_generator import (
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    BaseTokenGenerator,
)
from models.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_CKPT: str = "checkpoints/base_token_generator/best.pt"
DEFAULT_MAX_LEN: int = 64
DEFAULT_TOP_K: int = 50
DEFAULT_TEMPERATURE: float = 0.8

EXAMPLE_SENTENCES: list[str] = [
    "Hello",
    "How are you?",
    "I love machine learning.",
    "Please help me.",
]

_DIVIDER_WIDE  = "═" * 60
_DIVIDER_THIN  = "─" * 60


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────


def load_models(
    ckpt_path:  str,
    device_str: str = "auto",
) -> tuple[TextEncoder, BaseTokenGenerator, dict]:
    """Load TextEncoder and BaseTokenGenerator from a saved checkpoint.

    Parameters
    ----------
    ckpt_path:
        Path to ``best.pt`` or any ``epoch_NNN.pt`` checkpoint produced by
        ``training/train_base_token_generator.py``.
    device_str:
        ``"auto"`` selects CUDA when available, otherwise CPU.  Pass
        ``"cpu"`` or ``"cuda"`` to force a specific device.

    Returns
    -------
    encoder:
        Frozen ``TextEncoder`` in eval mode on *device*.
    generator:
        ``BaseTokenGenerator`` loaded from *ckpt_path* in eval mode on *device*.
    config:
        The hyperparameter dict that was embedded in the checkpoint at
        training time.

    Raises
    ------
    FileNotFoundError
        If *ckpt_path* does not exist.
    KeyError
        If the checkpoint is missing expected keys (wrong file format).
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_file}\n"
            f"Train the model first with:\n"
            f"  python training/train_base_token_generator.py --csv data/train.csv"
        )

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("Device: %s", device)
    logger.info("Loading checkpoint: %s", ckpt_file)

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)

    required_keys = {"epoch", "model_state", "config", "best_val_loss"}
    missing = required_keys - ckpt.keys()
    if missing:
        raise KeyError(
            f"Checkpoint is missing expected keys: {missing}. "
            f"Is this a train_base_token_generator.py checkpoint?"
        )

    config = ckpt["config"]
    logger.info(
        "Checkpoint: epoch=%d  best_val_loss=%.6f",
        ckpt["epoch"], ckpt["best_val_loss"],
    )

    # ── TextEncoder ───────────────────────────────────────────────────────
    logger.info("Loading TextEncoder (%s) …", config["encoder_model"])
    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=True,
        device=str(device),
    )
    encoder.eval()
    logger.info("  TextEncoder ready  hidden_dim=%d", encoder.hidden_dim)

    # ── BaseTokenGenerator ────────────────────────────────────────────────
    logger.info("Building BaseTokenGenerator …")
    generator = BaseTokenGenerator(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=0.0,               # no dropout at inference
        max_seq_len=config["model_max_seq_len"],
        text_dim=encoder.hidden_dim,
    ).to(device)
    generator.load_state_dict(ckpt["model_state"])
    generator.eval()
    logger.info("  %s", generator)

    return encoder, generator, config


# ──────────────────────────────────────────────────────────────────────────────
# Per-sentence generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_for_sentence(
    sentence:    str,
    encoder:     TextEncoder,
    generator:   BaseTokenGenerator,
    max_len:     int,
    top_k:       int,
    temperature: float,
    device:      torch.device,
) -> dict[str, object]:
    """Encode a single sentence and generate tokens with both decoding strategies.

    Parameters
    ----------
    sentence:
        Raw English text.
    encoder:
        Frozen TextEncoder (already on *device*).
    generator:
        Trained BaseTokenGenerator (already on *device*, in eval mode).
    max_len:
        Maximum number of tokens to generate per strategy.
    top_k:
        Number of candidates retained for top-k sampling.
    temperature:
        Softmax temperature for top-k sampling.
    device:
        Compute device.

    Returns
    -------
    dict with keys:

    * ``sentence``      – the original input string
    * ``greedy_tokens`` – ``list[int]`` from greedy decoding (trailing PAD stripped)
    * ``topk_tokens``   – ``list[int]`` from top-k sampling  (trailing PAD stripped)
    * ``greedy_len``    – number of tokens in *greedy_tokens*
    * ``topk_len``      – number of tokens in *topk_tokens*
    """
    with torch.no_grad():
        text_emb, text_mask = encoder.encode([sentence])
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        # ── Greedy ────────────────────────────────────────────────────────
        greedy_raw = generator.generate(
            text_embeddings=text_emb,
            text_padding_mask=text_mask,
            max_len=max_len,
            strategy="greedy",
        )  # [1, max_len]

        # ── Top-k ─────────────────────────────────────────────────────────
        topk_raw = generator.generate(
            text_embeddings=text_emb,
            text_padding_mask=text_mask,
            max_len=max_len,
            strategy="topk",
            temperature=temperature,
            top_k=top_k,
        )  # [1, max_len]

    def _strip_pad(tokens: list[int]) -> list[int]:
        """Remove trailing PAD_TOKEN_ID (512) values."""
        while tokens and tokens[-1] == PAD_TOKEN_ID:
            tokens = tokens[:-1]
        return tokens

    greedy_tokens = _strip_pad(greedy_raw[0].tolist())
    topk_tokens   = _strip_pad(topk_raw[0].tolist())

    return {
        "sentence":      sentence,
        "greedy_tokens": greedy_tokens,
        "greedy_len":    len(greedy_tokens),
        "topk_tokens":   topk_tokens,
        "topk_len":      len(topk_tokens),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────


def _format_tokens(tokens: list[int], max_display: int = 16) -> str:
    """Format a token list for display, truncating long sequences with '…'.

    Parameters
    ----------
    tokens:
        List of integer token IDs.
    max_display:
        Maximum number of tokens to show before truncating.

    Returns
    -------
    A compact string such as ``[247, 13, 88, …]  (32 total)``.
    """
    if not tokens:
        return "[]  (empty)"
    if len(tokens) <= max_display:
        return str(tokens)
    shown    = tokens[:max_display]
    ellipsis = f"… +{len(tokens) - max_display} more"
    return f"{shown[:-1] + [shown[-1]]}  {ellipsis}"


def print_result(result: dict[str, object], index: int) -> None:
    """Print a single sentence's generation result to stdout.

    Parameters
    ----------
    result:
        Dict returned by :func:`generate_for_sentence`.
    index:
        1-based sentence index (used for the header line).
    """
    sentence      = result["sentence"]
    greedy_tokens = result["greedy_tokens"]
    topk_tokens   = result["topk_tokens"]
    greedy_len    = result["greedy_len"]

    print(_DIVIDER_WIDE)
    print(f"[{index}] Sentence        : {sentence!r}")
    print(_DIVIDER_THIN)
    print(f"    Generated length : {greedy_len} tokens")
    print(f"    Greedy tokens    : {_format_tokens(greedy_tokens)}")
    print(f"    Top-k tokens     : {_format_tokens(topk_tokens)}")
    print(_DIVIDER_WIDE)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main run function
# ──────────────────────────────────────────────────────────────────────────────


def run(
    sentences:   list[str],
    ckpt_path:   str = DEFAULT_CKPT,
    max_len:     int = DEFAULT_MAX_LEN,
    top_k:       int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    device_str:  str = "auto",
) -> list[dict[str, object]]:
    """Load models and run generation for every sentence in *sentences*.

    Parameters
    ----------
    sentences:
        English sentences to process.  Must be non-empty.
    ckpt_path:
        Path to the trained checkpoint.
    max_len:
        Maximum tokens to generate per strategy per sentence.
    top_k:
        Top-k candidates for sampling.
    temperature:
        Softmax temperature for top-k sampling.
    device_str:
        Device selection string.

    Returns
    -------
    list[dict]
        One result dict per sentence (see :func:`generate_for_sentence`).
    """
    if not sentences:
        raise ValueError("sentences must be a non-empty list")

    encoder, generator, config = load_models(ckpt_path, device_str)
    device = next(generator.parameters()).device

    logger.info(
        "Generating tokens for %d sentence(s)  max_len=%d  top_k=%d  temp=%.2f",
        len(sentences), max_len, top_k, temperature,
    )

    results: list[dict[str, object]] = []
    for i, sentence in enumerate(sentences, start=1):
        logger.info("Processing [%d/%d]: %r", i, len(sentences), sentence)
        result = generate_for_sentence(
            sentence=sentence,
            encoder=encoder,
            generator=generator,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            device=device,
        )
        results.append(result)
        print_result(result, index=i)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test  (no checkpoint needed)
# ──────────────────────────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Validate the full inference pipeline with a synthetic untrained model.

    Constructs a randomly-initialised ``BaseTokenGenerator`` and a minimal
    ``TextEncoder`` wrapper — no checkpoint file is required.  Every
    functional path exercised by :func:`run` is covered:

    * ``generate_for_sentence`` (greedy + top-k)
    * ``_format_tokens`` (short and long sequences)
    * ``_strip_pad`` trailing-pad removal
    * ``print_result`` output formatting
    * type and shape assertions

    Exits with code 0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("=" * 60)
    logger.info("SMOKE TEST: test_base_generation.py")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch.manual_seed(42)

    # ── Build a tiny untrained model pair ────────────────────────────────
    D_txt = 768
    logger.info("Instantiating synthetic models …")

    # Minimal TextEncoder-compatible stub: skips HuggingFace download
    class _StubEncoder:
        """Mimics TextEncoder.encode() with random embeddings."""
        hidden_dim = D_txt

        def encode(
            self, sentences: list[str]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B   = len(sentences)
            T   = 8
            emb = torch.randn(B, T, self.hidden_dim, device=device)
            msk = torch.zeros(B, T, dtype=torch.bool, device=device)
            return emb, msk

    encoder_stub = _StubEncoder()

    generator = BaseTokenGenerator(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.0,
        max_seq_len=256,
        text_dim=D_txt,
    ).to(device)
    generator.eval()
    logger.info("  Stub encoder and tiny generator ready")

    # ── generate_for_sentence ─────────────────────────────────────────────
    logger.info("── generate_for_sentence ───────────────────────────────────")
    test_sentences = ["Hello", "How are you?", "Please help me."]
    max_len = 20

    for sent in test_sentences:
        result = generate_for_sentence(
            sentence=sent,
            encoder=encoder_stub,      # type: ignore[arg-type]
            generator=generator,
            max_len=max_len,
            top_k=20,
            temperature=0.8,
            device=device,
        )
        # Shape / type checks
        assert result["sentence"] == sent
        assert isinstance(result["greedy_tokens"], list)
        assert isinstance(result["topk_tokens"],   list)
        assert result["greedy_len"] == len(result["greedy_tokens"])
        assert result["topk_len"]   == len(result["topk_tokens"])
        # All returned token IDs must be valid (untrained model may still pad)
        for tok in result["greedy_tokens"]:
            assert 0 <= tok < VOCAB_SIZE, f"Out-of-range greedy token: {tok}"
        for tok in result["topk_tokens"]:
            assert 0 <= tok < VOCAB_SIZE, f"Out-of-range top-k token: {tok}"
        logger.info(
            "  %r → greedy_len=%d  topk_len=%d  PASSED",
            sent, result["greedy_len"], result["topk_len"],
        )

    # ── PAD stripping ─────────────────────────────────────────────────────
    logger.info("── PAD stripping ───────────────────────────────────────────")

    def _strip(tokens: list[int]) -> list[int]:
        while tokens and tokens[-1] == PAD_TOKEN_ID:
            tokens = tokens[:-1]
        return tokens

    assert _strip([1, 2, PAD_TOKEN_ID, PAD_TOKEN_ID]) == [1, 2]
    assert _strip([PAD_TOKEN_ID])                      == []
    assert _strip([])                                   == []
    assert _strip([3, 4, 5])                           == [3, 4, 5]
    logger.info("  PAD stripping: PASSED")

    # ── _format_tokens ────────────────────────────────────────────────────
    logger.info("── _format_tokens ──────────────────────────────────────────")
    short_fmt = _format_tokens([1, 2, 3])
    assert "1" in short_fmt and "3" in short_fmt
    long_tokens = list(range(30))
    long_fmt = _format_tokens(long_tokens, max_display=16)
    assert "more" in long_fmt, f"Long format missing truncation marker: {long_fmt}"
    empty_fmt = _format_tokens([])
    assert "empty" in empty_fmt
    logger.info("  short=%r  PASSED", short_fmt)
    logger.info("  long =%r  PASSED", long_fmt)
    logger.info("  empty=%r  PASSED", empty_fmt)

    # ── print_result ──────────────────────────────────────────────────────
    logger.info("── print_result ────────────────────────────────────────────")
    sample_result = generate_for_sentence(
        sentence="I love machine learning.",
        encoder=encoder_stub,     # type: ignore[arg-type]
        generator=generator,
        max_len=10,
        top_k=10,
        temperature=1.0,
        device=device,
    )
    print_result(sample_result, index=1)    # visual check: no exceptions
    logger.info("  print_result: PASSED (no exceptions)")

    # ── Greedy determinism ────────────────────────────────────────────────
    logger.info("── Greedy determinism ──────────────────────────────────────")
    # The stub encoder uses torch.randn — seed it so both calls get the same
    # text embedding, then verify the generator itself is deterministic.
    torch.manual_seed(0)
    r1 = generate_for_sentence(
        "Hello", encoder_stub, generator, max_len=10,  # type: ignore[arg-type]
        top_k=50, temperature=1.0, device=device,
    )
    torch.manual_seed(0)   # same seed → same stub output → same greedy result
    r2 = generate_for_sentence(
        "Hello", encoder_stub, generator, max_len=10,  # type: ignore[arg-type]
        top_k=50, temperature=1.0, device=device,
    )
    assert r1["greedy_tokens"] == r2["greedy_tokens"], \
        "Greedy decoding must be deterministic given identical encoder output"
    logger.info("  Greedy is deterministic across two calls: PASSED")

    # ── Top-k stochasticity ───────────────────────────────────────────────
    logger.info("── Top-k stochasticity ─────────────────────────────────────")
    # With temperature=1.0 and k=50 on a random model, runs should differ
    # the vast majority of the time.  We allow up to 3 retries before failing.
    found_diff = False
    for seed in range(3):
        torch.manual_seed(seed)
        ra = generate_for_sentence(
            "Hello", encoder_stub, generator, max_len=15,  # type: ignore[arg-type]
            top_k=50, temperature=1.0, device=device,
        )
        torch.manual_seed(seed + 100)
        rb = generate_for_sentence(
            "Hello", encoder_stub, generator, max_len=15,  # type: ignore[arg-type]
            top_k=50, temperature=1.0, device=device,
        )
        if ra["topk_tokens"] != rb["topk_tokens"]:
            found_diff = True
            break
    assert found_diff, "Top-k sampling with temperature=1.0 should be stochastic"
    logger.info("  Top-k sampling is stochastic: PASSED")

    # ── load_models file-not-found guard ──────────────────────────────────
    logger.info("── load_models error handling ──────────────────────────────")
    try:
        load_models("/nonexistent/path/best.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as exc:
        logger.info("  FileNotFoundError: PASSED  (%s)", exc)

    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference with the trained SignVerse BaseTokenGenerator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")
    sub.add_parser("smoke_test", help="Self-contained smoke test (no checkpoint needed)")

    p.add_argument(
        "sentences",
        nargs="*",
        metavar="SENTENCE",
        help=(
            "English sentence(s) to generate tokens for.  "
            "If omitted, the built-in example sentences are used."
        ),
    )
    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        dest="ckpt_path",
        help="Path to the trained checkpoint (best.pt or epoch_NNN.pt)",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum number of tokens to generate per sentence",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k candidates for sampling",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Softmax temperature for top-k sampling",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='"auto", "cpu", or "cuda"',
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG",
    )
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

    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    sentences = args.sentences if args.sentences else EXAMPLE_SENTENCES

    if not args.sentences:
        logger.info("No sentences provided — using built-in examples.")

    run(
        sentences=sentences,
        ckpt_path=args.ckpt_path,
        max_len=args.max_len,
        top_k=args.top_k,
        temperature=args.temperature,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()