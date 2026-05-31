"""
models/text_encoder.py
SignVerse – Text encoder based on google/flan-t5-base.

Wraps the T5 encoder stack (no decoder) to produce contextualised token
embeddings from English sentences. The output is used as cross-attention
conditioning for the downstream motion-generation model.

Architecture notes (flan-t5-base)
----------------------------------
    d_model (hidden size) : 768
    num_encoder_layers    : 12
    num_attention_heads   : 12
    vocab_size            : 32,128  (SentencePiece, shared with decoder)
    max_position_ids      : 512

Output
------
    Tensor[B, T_text, 768]  – one 768-d vector per input token, including the
    appended </s> (EOS) token that T5 adds automatically.  Padding positions
    are zeroed out before return so downstream cross-attention can use the
    accompanying key_padding_mask directly.

Usage
-----
    encoder = TextEncoder(device="cuda")
    embeddings, mask = encoder(["Hello world", "Sign language is visual."])
    # embeddings : [2, T_max, 768]
    # mask       : [2, T_max]  True where padding
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID: str = "google/flan-t5-base"
HIDDEN_DIM: int = 768          # d_model for flan-t5-base
MAX_TOKEN_LEN: int = 512       # hard cap imposed by T5 position embeddings


# ── TextEncoder ────────────────────────────────────────────────────────────────


class TextEncoder(nn.Module):
    """Frozen (or fine-tunable) T5 encoder that maps English text to embeddings.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier or local directory path.
        Defaults to ``"google/flan-t5-base"``.
    max_length:
        Maximum number of sub-word tokens per sentence (including the EOS
        token that T5 appends).  Sequences longer than this are truncated.
        Must be ≤ 512 for flan-t5-base.
    freeze:
        When ``True`` (default) all encoder parameters are frozen.  The
        encoder acts as a fixed feature extractor, which is the standard
        approach when the downstream model is much smaller than the encoder.
        Set to ``False`` to allow end-to-end fine-tuning.
    device:
        ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"auto"`` (selects CUDA if
        available, CPU otherwise).  The encoder and all tensors it produces
        are kept on this device.
    torch_dtype:
        Floating-point precision for the encoder weights.  Defaults to
        ``torch.float32``.  Use ``torch.float16`` or ``torch.bfloat16`` for
        memory-efficient inference on GPU.
    """

    def __init__(
        self,
        model_name_or_path: str = MODEL_ID,
        max_length: int = 128,
        freeze: bool = True,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_length = min(max_length, MAX_TOKEN_LEN)
        self.freeze = freeze
        self.torch_dtype = torch_dtype

        # ── Device resolution ─────────────────────────────────────────────
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info(
            "TextEncoder: loading %s on %s (freeze=%s, dtype=%s)",
            model_name_or_path,
            self._device,
            freeze,
            torch_dtype,
        )

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            model_max_length=self.max_length,
        )

        # ── Encoder (T5 encoder stack only; no decoder weights loaded) ────
        self.encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(self._device)

        # ── Optionally freeze all encoder parameters ──────────────────────
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()
            logger.info("TextEncoder: all encoder parameters frozen")

        # Exposed for downstream models
        self.hidden_dim: int = self.encoder.config.d_model  # 768 for flan-t5-base

    # ── Tokenization ──────────────────────────────────────────────────────────

    def tokenize(
        self,
        sentences: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenise a batch of sentences and move tensors to the encoder device.

        Parameters
        ----------
        sentences:
            A list of raw English strings.  Whitespace is normalised by the
            SentencePiece tokenizer; no pre-processing is required by the
            caller.

        Returns
        -------
        dict with keys ``input_ids`` and ``attention_mask``, both
        ``LongTensor[B, T]`` on ``self._device``.
        """
        if not sentences:
            raise ValueError("sentences must be a non-empty list of strings.")
        if any(not isinstance(s, str) for s in sentences):
            raise TypeError(
                "All elements of sentences must be str. "
                f"Got types: {[type(s).__name__ for s in sentences]}"
            )

        encoding = self.tokenizer(
            sentences,
            padding=True,           # pad to longest in batch
            truncation=True,        # truncate to self.max_length
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding.input_ids.to(self._device),
            "attention_mask": encoding.attention_mask.to(self._device),
        }

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        sentences: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of sentences without gradient tracking.

        Convenience wrapper around :meth:`forward` that disables gradient
        computation.  Use this at inference time or when building conditioning
        tensors outside the training loop.

        Parameters
        ----------
        sentences:
            A list of raw English strings.

        Returns
        -------
        embeddings : ``FloatTensor[B, T, D]``
            Token-level embeddings.  Padding positions are zero-filled.
        padding_mask : ``BoolTensor[B, T]``
            ``True`` at positions that correspond to padding tokens.
            Compatible with ``nn.MultiheadAttention(key_padding_mask=...)``.
        """
        with torch.no_grad():
            return self.forward(sentences)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        sentences: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: tokenise → encode → mask padding positions.

        Parameters
        ----------
        sentences:
            A list of raw English strings (batch size B).

        Returns
        -------
        embeddings : ``FloatTensor[B, T, D]``
            Contextualised token embeddings from the final encoder layer.
            Shape: ``[B, T_max, hidden_dim]`` where ``T_max`` is the length
            of the longest sentence in the batch after tokenisation.
            Padding positions are explicitly zeroed so they carry no signal.
        padding_mask : ``BoolTensor[B, T]``
            ``True`` at padding positions (matches PyTorch's
            ``key_padding_mask`` convention for ``nn.MultiheadAttention``).

        Notes
        -----
        * Batch-internal dynamic padding: ``T_max`` varies per batch — no
          fixed-length overhead from shorter batches.
        * When ``freeze=True`` the forward pass runs under ``torch.no_grad()``
          automatically (parameters have ``requires_grad=False``).
        * Mixed precision: if the encoder was loaded with ``torch_dtype=
          torch.float16``, embeddings are returned in fp16.  Cast in the
          caller if your downstream model expects fp32.
        """
        tokenized = self.tokenize(sentences)
        input_ids:      torch.Tensor = tokenized["input_ids"]       # [B, T]
        attention_mask: torch.Tensor = tokenized["attention_mask"]  # [B, T]

        # T5EncoderModel.forward returns a BaseModelOutput
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        embeddings: torch.Tensor = encoder_output.last_hidden_state  # [B, T, D]

        # Zero-fill padding positions so they are inert for any downstream
        # operation that does not use the mask (e.g. mean pooling, concat).
        # attention_mask: 1 = real token, 0 = padding  →  broadcast [B, T, 1]
        mask_expanded = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        embeddings = embeddings * mask_expanded

        # padding_mask: True where padding (PyTorch key_padding_mask convention)
        padding_mask: torch.Tensor = attention_mask.eq(0)  # [B, T]

        return embeddings, padding_mask

    # ── Utilities ─────────────────────────────────────────────────────────────

    def to(self, device: torch.device | str, **kwargs) -> "TextEncoder":  # type: ignore[override]
        """Move encoder and internal device tracker together."""
        self._device = torch.device(device)
        self.encoder = self.encoder.to(device, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        """The device this encoder currently lives on."""
        return self._device

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total (or trainable-only) parameter count."""
        params = (
            self.encoder.parameters()
            if not trainable_only
            else filter(lambda p: p.requires_grad, self.encoder.parameters())
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        return (
            f"TextEncoder("
            f"model={self.model_name_or_path!r}, "
            f"hidden_dim={self.hidden_dim}, "
            f"max_length={self.max_length}, "
            f"freeze={self.freeze}, "
            f"device={self._device}, "
            f"params={self.num_parameters():,}"
            f")"
        )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID

    print(f"\nLoading TextEncoder from: {model_path}")
    encoder = TextEncoder(
        model_name_or_path=model_path,
        max_length=128,
        freeze=True,
        device="auto",
    )
    print(encoder)
    print(f"  Trainable params : {encoder.num_parameters(trainable_only=True):,}")
    print(f"  Total params     : {encoder.num_parameters():,}")
    print()

    # ── Tokenisation check ─────────────────────────────────────────────────
    sentences = [
        "Don't keep me on tenterhooks!",
        "I like to be home, that way I can work on my story.",
        "Can I stay here tonight?",
        "The temperature is below zero today.",
    ]

    print("── Tokenisation ─────────────────────────────────────────────")
    tokenized = encoder.tokenize(sentences)
    print(f"  input_ids shape      : {tuple(tokenized['input_ids'].shape)}")
    print(f"  attention_mask shape : {tuple(tokenized['attention_mask'].shape)}")
    print(f"  device               : {tokenized['input_ids'].device}")
    print()

    # ── Encoding check ─────────────────────────────────────────────────────
    print("── Encoding (no_grad) ───────────────────────────────────────")
    embeddings, padding_mask = encoder.encode(sentences)
    B, T, D = embeddings.shape
    print(f"  embeddings shape     : {(B, T, D)}  (B, T_max, hidden_dim)")
    print(f"  padding_mask shape   : {tuple(padding_mask.shape)}")
    print(f"  embeddings dtype     : {embeddings.dtype}")
    print(f"  embeddings device    : {embeddings.device}")
    print()

    # ── Per-sample lengths ─────────────────────────────────────────────────
    print("── Per-sample token lengths ─────────────────────────────────")
    for i, sentence in enumerate(sentences):
        real_len = (~padding_mask[i]).sum().item()
        print(f"  [{i}] len={real_len:>3}  {sentence[:60]}")
    print()

    # ── Padding position check ─────────────────────────────────────────────
    print("── Padding mask verification ────────────────────────────────")
    pad_positions = padding_mask.sum().item()
    total_positions = padding_mask.numel()
    print(f"  Padding positions    : {int(pad_positions)}/{total_positions} "
          f"({100*pad_positions/total_positions:.1f}%)")

    # Verify zeroing: padded positions in embeddings should be all-zero
    pad_norms = embeddings[padding_mask].norm(dim=-1)
    assert pad_norms.max().item() == 0.0, "Padding positions must be zeroed!"
    print("  Zero-fill check      : PASSED (all padding positions are zero)")
    print()

    # ── Single-sentence edge case ──────────────────────────────────────────
    print("── Single-sentence batch ────────────────────────────────────")
    emb_single, mask_single = encoder.encode(["Where is the nearest drugstore?"])
    print(f"  shape : {tuple(emb_single.shape)}")
    print(f"  mask  : {tuple(mask_single.shape)}")
    print()

    # ── Forward (with grad) check ──────────────────────────────────────────
    print("── forward() gradient flow ──────────────────────────────────")
    emb_fwd, _ = encoder.forward(sentences[:2])
    print(f"  requires_grad : {emb_fwd.requires_grad}  (expected False when freeze=True)")
    print(f"  shape         : {tuple(emb_fwd.shape)}")
    print()

    print("All smoke tests passed.")
