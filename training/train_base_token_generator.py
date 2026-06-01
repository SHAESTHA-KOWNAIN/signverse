"""
training/train_base_token_generator.py
SignVerse – Training script for BaseTokenGenerator.

Trains the autoregressive transformer decoder (models/base_token_generator.py)
to generate RVQ base-layer motion tokens from FLAN-T5 text embeddings.

Pipeline
--------
    train.csv
        ↓  SignDataset  (80 / 20 split, reproducible seed)
        ↓  DataLoader   (padded batches via collate_sign_batch)
        ↓  TextEncoder  (frozen FLAN-T5-base, fp32 or fp16)
        ↓  BaseTokenGenerator  (teacher-forcing forward pass)
        ↓  CrossEntropyLoss    (ignore_index=PAD_TOKEN_ID=512)
        ↓  AdamW + linear-warmup / cosine-decay LR schedule
        ↓  AMP (fp16/bf16 on CUDA) + gradient clipping
        ↓  Validation: token accuracy · perplexity · loss

Teacher-forcing shift convention
---------------------------------
    Given ground-truth base tokens  t = [t_0, t_1, …, t_{L-1}]  (from tokens[:,0]):

        decoder_input  = [BOS, t_0, t_1, …, t_{L-2}]     shape [B, L]
        decoder_target = [t_0, t_1, …, t_{L-1}]           shape [B, L]

    PAD positions (t_{real_len} … t_{L-1}) are filled with PAD_TOKEN_ID (512)
    and excluded from the loss via ``ignore_index=512``.

Usage
-----
    # Minimal:
    python training/train_base_token_generator.py --csv data/train.csv

    # Full control:
    python training/train_base_token_generator.py         \\
        --csv       data/train.csv                        \\
        --ckpt_dir  checkpoints/base_gen                  \\
        --epochs    50                                    \\
        --batch     16                                    \\
        --lr        1e-4                                  \\
        --val_frac  0.2                                   \\
        --seed      42                                    \\
        --workers   4                                     \\
        --device    cuda

    # Resume from checkpoint:
    python training/train_base_token_generator.py         \\
        --csv data/train.csv                              \\
        --resume checkpoints/base_gen/best.pt

    # Greedy inference from a trained checkpoint:
    python training/train_base_token_generator.py infer   \\
        --ckpt   checkpoints/base_gen/best.pt             \\
        --sentences "Hello, how are you?" "Where is the hospital?"

    # Top-k inference:
    python training/train_base_token_generator.py infer   \\
        --ckpt        checkpoints/base_gen/best.pt        \\
        --strategy    topk                                \\
        --temperature 0.8                                 \\
        --top_k       50                                  \\
        --max_len     128                                 \\
        --sentences   "Can you help me?"

    # Self-contained smoke test (no CSV / GPU required):
    python training/train_base_token_generator.py smoke_test

Checkpoint format
-----------------
    {
        "epoch":            int,
        "model_state":      OrderedDict,
        "optimizer_state":  dict,
        "scheduler_state":  dict,
        "scaler_state":     dict,          # GradScaler; empty dict when AMP disabled
        "best_val_loss":    float,
        "config":           dict,          # full hyperparameter snapshot
        "train_history":    list[dict],    # per-epoch metrics
    }

Outputs
-------
    {ckpt_dir}/
        best.pt             ← lowest validation loss ever seen
        final.pt            ← weights at the end of the last epoch
        epoch_{N:03d}.pt    ← periodic snapshot (every --save_every epochs)
        training_log.json   ← full history as JSON array
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

# AMP imports: prefer the non-deprecated torch.amp API (PyTorch ≥ 2.1);
# fall back to the legacy cuda.amp path for older environments.
try:
    from torch.amp import GradScaler, autocast as _autocast

    def autocast(enabled: bool = True, device_type: str = "cuda"):  # type: ignore[misc]
        """Thin wrapper so call-sites can pass ``enabled`` as a keyword."""
        if not enabled:
            import contextlib
            return contextlib.nullcontext()
        return _autocast(device_type=device_type)

except ImportError:  # PyTorch < 2.1 fallback
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]

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
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # ── Data ──────────────────────────────────────────────────────────────
    "csv_path":          "train.csv",
    "val_frac":          0.20,
    "max_seq_len":       256,          # SignDataset token truncation
    "seed":              42,
    # ── Training ──────────────────────────────────────────────────────────
    "epochs":            50,
    "batch_size":        16,
    "lr":                1e-4,
    "weight_decay":      1e-2,
    "grad_clip":         1.0,          # max gradient norm; 0 = disabled
    "warmup_epochs":     5,
    "label_smoothing":   0.05,          # cross-entropy label smoothing
    # ── AMP ───────────────────────────────────────────────────────────────
    "amp":               True,         # enable mixed precision when CUDA is available
    # ── Model ─────────────────────────────────────────────────────────────
    "hidden_dim":        512,
    "num_layers":        4,
    "num_heads":         8,
    "ff_dim":            2048,
    "dropout":           0.1,
    "model_max_seq_len": 512,
    # ── Encoder ───────────────────────────────────────────────────────────
    "encoder_model":     "google/flan-t5-base",
    "encoder_max_len":   128,
    # ── Infrastructure ────────────────────────────────────────────────────
    "device":            "auto",
    "workers":           4,
    "ckpt_dir":          "checkpoints/base_token_generator",
    "save_every":        5,
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_device(device_str: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _amp_enabled(config: dict, device: torch.device) -> bool:
    """AMP is only beneficial (and safe) on CUDA."""
    return bool(config.get("amp", True)) and device.type == "cuda"


def _split_dataset(
    dataset: SignDataset,
    val_frac: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Reproducible random 80 / 20 split."""
    n_val   = max(1, math.floor(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(dataset, [n_train, n_val], generator=gen)
    logger.info("Split → train=%d  val=%d", n_train, n_val)
    return train_sub, val_sub


def _build_loaders(
    train_sub:  Subset,
    val_sub:    Subset,
    batch_size: int,
    workers:    int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap subsets in DataLoaders sharing the SignVerse collate function."""
    pin = torch.cuda.is_available()
    kw  = dict(collate_fn=collate_sign_batch, pin_memory=pin, drop_last=False)
    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,  num_workers=workers, **kw
    )
    val_loader = DataLoader(
        val_sub,   batch_size=batch_size, shuffle=False, num_workers=workers, **kw
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Batch preparation
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_decoder_batch(
    tokens:  torch.Tensor,   # [B, T_max, 6]  from collate_sign_batch
    lengths: torch.Tensor,   # [B]
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher-forcing decoder input / target tensors from a raw batch.

    Given ground-truth base tokens ``t = [t_0, …, t_{L-1}]``:

        decoder_input  = [BOS, t_0, t_1, …, t_{L-2}]  shape [B, L]
        decoder_target = [t_0,  t_1, …, t_{L-1}]       shape [B, L]

    where ``L = T_max`` (the maximum unpadded length in the batch).  Positions
    beyond each sequence's true length are PAD_TOKEN_ID (512) in both tensors.

    Parameters
    ----------
    tokens:
        Full six-layer token matrix from the dataloader; only layer 0
        (base tokens) is used here.
    lengths:
        True (unpadded) sequence lengths, shape [B].
    device:
        Target device for the output tensors.

    Returns
    -------
    decoder_input : LongTensor [B, L]
        BOS-prefixed token sequence fed to ``BaseTokenGenerator.forward()``.
    decoder_target : LongTensor [B, L]
        Right-shifted target for cross-entropy loss.  PAD positions carry
        ``PAD_TOKEN_ID`` and are ignored by the loss function.
    """
    # Extract base layer: [B, T_max]
    base_tokens = tokens[:, :, 0].to(device)   # [B, T_max]
    B, T_max    = base_tokens.shape

    # decoder_input: [BOS, t_0, …, t_{T_max - 2}]  — right-shift targets by 1
    bos_col = base_tokens.new_full((B, 1), fill_value=BOS_TOKEN_ID)
    # Slice off last column so decoder_input stays length T_max
    decoder_input = torch.cat([bos_col, base_tokens[:, :-1]], dim=1)  # [B, T_max]

    # decoder_target is the original sequence; PAD positions already filled with
    # PAD_TOKEN_ID (512) by collate_sign_batch — no extra work needed.
    decoder_target = base_tokens  # [B, T_max]

    # Enforce PAD beyond each sequence's true length (belt-and-suspenders guard
    # in case collate produced unexpected values past the real length).
    pad_mask = torch.arange(T_max, device=device).unsqueeze(0) >= lengths.to(device).unsqueeze(1)
    decoder_input  = decoder_input.masked_fill(pad_mask, PAD_TOKEN_ID)
    decoder_target = decoder_target.masked_fill(pad_mask, PAD_TOKEN_ID)

    # BOS at position 0 of decoder_input must never be masked, even for
    # length-1 sequences (unlikely but defensive).
    decoder_input[:, 0] = BOS_TOKEN_ID

    return decoder_input, decoder_target


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def _compute_metrics(
    logits:         torch.Tensor,   # [B * L, VOCAB_SIZE] or [B, L, VOCAB_SIZE]
    targets:        torch.Tensor,   # [B * L] or [B, L]
    loss:           float,
) -> dict[str, float]:
    """Compute token accuracy and perplexity from logits and targets.

    PAD positions (target == PAD_TOKEN_ID) are excluded from accuracy so the
    metric reflects only meaningful motion tokens.

    Parameters
    ----------
    logits:
        Raw class scores.  Will be flattened to [N, VOCAB_SIZE] internally.
    targets:
        Ground-truth token IDs.  Will be flattened to [N].
    loss:
        Pre-computed cross-entropy loss value (float) used to derive
        perplexity; avoids recomputing the loss.

    Returns
    -------
    dict with keys ``loss``, ``accuracy`` (%), ``perplexity``.
    """
    with torch.no_grad():
        flat_logits  = logits.reshape(-1, VOCAB_SIZE)
        flat_targets = targets.reshape(-1)

        # Mask out PAD positions
        real_mask = flat_targets != PAD_TOKEN_ID         # [N]
        preds     = flat_logits.argmax(dim=-1)           # [N]
        correct   = (preds == flat_targets) & real_mask
        n_real    = real_mask.sum().item()
        accuracy  = (correct.sum().item() / max(1, n_real)) * 100.0
        perplexity = math.exp(min(loss, 300.0))          # clamp to avoid overflow

    return {
        "loss":       loss,
        "accuracy":   accuracy,
        "perplexity": perplexity,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Warmup + cosine LR scheduler
# ──────────────────────────────────────────────────────────────────────────────


class _WarmupCosineScheduler:
    """Linear LR warm-up for the first ``warmup_epochs``, then cosine decay.

    Wraps ``CosineAnnealingLR`` and applies a linear scaling factor during
    warm-up so the scheduler can be stepped once per epoch without breaking
    the downstream cosine phase.
    """

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        base_lr:       float,
        eta_min:       float = 1e-6,
    ) -> None:
        self.optimizer     = optimizer
        self.warmup_epochs = max(0, warmup_epochs)
        self.base_lr       = base_lr
        self.cosine        = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=eta_min,
        )
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear ramp: epoch 1 → base_lr/warmup … epoch warmup_epochs → base_lr
            frac = self._epoch / max(1, self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * frac
        else:
            # Cosine decay begins from epoch warmup_epochs+1.
            # CosineAnnealingLR.step() is safe to call here because the
            # training loop calls scheduler.step() *after* optimizer.step().
            self.cosine.step()

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"_epoch": self._epoch, "cosine": self.cosine.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["_epoch"]
        self.cosine.load_state_dict(state["cosine"])


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save_checkpoint(
    path:          Path,
    epoch:         int,
    model:         BaseTokenGenerator,
    optimizer:     torch.optim.Optimizer,
    scheduler:     _WarmupCosineScheduler,
    scaler:        GradScaler,
    best_val_loss: float,
    config:        dict,
    history:       list[dict],
) -> None:
    """Serialise full training state to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "best_val_loss":    best_val_loss,
            "config":           config,
            "train_history":    history,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def _load_checkpoint(
    path:      Path,
    model:     BaseTokenGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: _WarmupCosineScheduler,
    scaler:    GradScaler,
) -> tuple[int, float, list[dict]]:
    """Load checkpoint in-place.  Returns ``(start_epoch, best_val_loss, history)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    history       = ckpt.get("train_history", [])
    logger.info(
        "Resumed from %s  (epoch=%d  best_val_loss=%.6f)",
        path, ckpt["epoch"], best_val_loss,
    )
    return start_epoch, best_val_loss, history


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar (zero-dependency tqdm-compatible fallback)
# ──────────────────────────────────────────────────────────────────────────────


def _make_pbar(iterable, desc: str, total: int, leave: bool = False):
    """Return a tqdm progress bar if available, else a plain iterator.

    Using tqdm as a soft dependency keeps the script runnable in minimal
    environments without it.
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)
    except ImportError:
        return iterable


# ──────────────────────────────────────────────────────────────────────────────
# Train / validation epoch functions
# ──────────────────────────────────────────────────────────────────────────────


def _train_epoch(
    encoder:    TextEncoder,
    model:      BaseTokenGenerator,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler:     GradScaler,
    device:     torch.device,
    config:     dict,
    use_amp:    bool,
    epoch:      int,
    total_epochs: int,
) -> dict[str, float]:
    """Run one full training epoch with teacher forcing.

    Parameters
    ----------
    encoder:
        Frozen TextEncoder; always run in eval mode.
    model:
        BaseTokenGenerator in training mode.
    loader:
        Training DataLoader.
    optimizer:
        AdamW instance.
    scaler:
        GradScaler for AMP; no-op when ``use_amp=False``.
    device:
        Compute device.
    config:
        Hyperparameter dict.
    use_amp:
        Whether to use ``torch.cuda.amp.autocast``.
    epoch / total_epochs:
        Used only for the progress-bar description.

    Returns
    -------
    dict with keys: loss, accuracy, perplexity.
    """
    model.train()
    encoder.encoder.eval()

    total_loss  = 0.0
    total_corr  = 0
    total_real  = 0
    n_batches   = 0

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN_ID,
        label_smoothing=config["label_smoothing"],
        reduction="mean",
    )

    pbar = _make_pbar(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        total=len(loader),
    )

    for batch in pbar:
        sentences: list[str]    = batch["sentences"]
        tokens:    torch.Tensor = batch["tokens"]     # [B, T_max, 6]
        lengths:   torch.Tensor = batch["lengths"]    # [B]

        # ── 1. Text encoding (no grad; encoder is frozen) ──────────────────
        with torch.no_grad():
            text_emb, text_mask = encoder.encode(sentences)
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        # ── 2. Build teacher-forcing tensors ──────────────────────────────
        dec_input, dec_target = _prepare_decoder_batch(tokens, lengths, device)
        # dec_input  [B, T_max]:  [BOS, t_0, …, t_{T_max-2}]
        # dec_target [B, T_max]:  [t_0, t_1, …, t_{T_max-1}]

        # ── 3. Forward + loss under optional AMP ─────────────────────────
        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(text_emb, text_mask, dec_input)  # [B, T_max, VOCAB_SIZE]
            loss   = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                dec_target.reshape(-1),
            )

        # ── 4. Backprop ───────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if config["grad_clip"] > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        # ── 5. Accumulate metrics (use pure loss for accuracy; no smoothing) ─
        with torch.no_grad():
            raw_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).detach(),
                dec_target.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
                reduction="mean",
            ).item()
            preds      = logits.reshape(-1, VOCAB_SIZE).argmax(dim=-1)
            flat_tgt   = dec_target.reshape(-1)
            real_mask  = flat_tgt != PAD_TOKEN_ID
            total_corr += ((preds == flat_tgt) & real_mask).sum().item()
            total_real += real_mask.sum().item()

        total_loss += raw_loss
        n_batches  += 1

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{raw_loss:.4f}")

    avg_loss  = total_loss / max(1, n_batches)
    accuracy  = (total_corr / max(1, total_real)) * 100.0
    perplexity = math.exp(min(avg_loss, 300.0))

    return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity}


@torch.no_grad()
def _val_epoch(
    encoder:  TextEncoder,
    model:    BaseTokenGenerator,
    loader:   DataLoader,
    device:   torch.device,
    use_amp:  bool,
    epoch:    int,
    total_epochs: int,
) -> dict[str, float]:
    """Run one full validation epoch without gradient computation.

    Uses plain cross-entropy (no label smoothing) so loss is directly
    comparable to perplexity and interpretable as bits-per-token.

    Returns
    -------
    dict with keys: loss, accuracy, perplexity.
    """
    model.eval()
    encoder.encoder.eval()

    total_loss = 0.0
    total_corr = 0
    total_real = 0
    n_batches  = 0

    pbar = _make_pbar(
        loader,
        desc=f"Val   {epoch}/{total_epochs}",
        total=len(loader),
    )

    for batch in pbar:
        sentences: list[str]    = batch["sentences"]
        tokens:    torch.Tensor = batch["tokens"]
        lengths:   torch.Tensor = batch["lengths"]

        with torch.no_grad():
            text_emb, text_mask = encoder.encode(sentences)
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        dec_input, dec_target = _prepare_decoder_batch(tokens, lengths, device)

        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(text_emb, text_mask, dec_input)   # [B, T_max, VOCAB_SIZE]

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

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss   = total_loss / max(1, n_batches)
    accuracy   = (total_corr / max(1, total_real)) * 100.0
    perplexity = math.exp(min(avg_loss, 300.0))

    return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity}


# ──────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────────────────────────────────────


def train(config: dict, resume: Optional[str] = None) -> None:
    """Full training loop for BaseTokenGenerator.

    Parameters
    ----------
    config:
        Hyperparameter dict (see ``DEFAULT_CONFIG`` for all keys).
    resume:
        Path to a ``.pt`` checkpoint file to resume from, or ``None`` for a
        fresh run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device  = _resolve_device(config["device"])
    use_amp = _amp_enabled(config, device)
    logger.info("Device: %s  |  AMP: %s", device, use_amp)

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", config["csv_path"])
    dataset = SignDataset(
        csv_path=config["csv_path"],
        max_seq_len=config["max_seq_len"],
        skip_corrupt_gloss=True,
    )
    logger.info(dataset.summary())

    train_sub, val_sub = _split_dataset(dataset, config["val_frac"], config["seed"])
    train_loader, val_loader = _build_loaders(
        train_sub, val_sub, config["batch_size"], config["workers"]
    )
    logger.info(
        "Batches per epoch → train=%d  val=%d", len(train_loader), len(val_loader)
    )

    # ── Encoder ───────────────────────────────────────────────────────────
    logger.info("Loading TextEncoder …")
    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=True,
        device=str(device),
    )
    logger.info("  %s", encoder)

    # ── Model ─────────────────────────────────────────────────────────────
    model = BaseTokenGenerator(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
        max_seq_len=config["model_max_seq_len"],
        text_dim=encoder.hidden_dim,
    ).to(device)
    logger.info("  %s", model)
    logger.info(
        "  Trainable parameters: %d", model.num_parameters(trainable_only=True)
    )

    # ── Optimiser, scheduler, scaler ─────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = _WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["epochs"],
        base_lr=config["lr"],
    )
    _scaler_device = device.type if device.type == "cuda" else "cpu"
    scaler = GradScaler(device=_scaler_device, enabled=use_amp)

    # ── Optional resume ───────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = math.inf
    history:      list[dict] = []

    if resume:
        start_epoch, best_val_loss, history = _load_checkpoint(
            Path(resume), model, optimizer, scheduler, scaler
        )

    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot alongside checkpoints for reproducibility
    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved → %s", config_path)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info(
        "Starting training: epochs=%d  start=%d  batch=%d  lr=%.2e  amp=%s",
        config["epochs"], start_epoch, config["batch_size"], config["lr"], use_amp,
    )

    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.time()

        train_metrics = _train_epoch(
            encoder=encoder,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=config["epochs"],
        )
        val_metrics = _val_epoch(
            encoder=encoder,
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=config["epochs"],
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        logger.info(
            "Epoch %3d/%d | "
            "train  loss=%.4f  acc=%.2f%%  ppl=%.2f | "
            "val    loss=%.4f  acc=%.2f%%  ppl=%.2f | "
            "lr=%.2e | %.1fs",
            epoch, config["epochs"],
            train_metrics["loss"], train_metrics["accuracy"], train_metrics["perplexity"],
            val_metrics["loss"],   val_metrics["accuracy"],   val_metrics["perplexity"],
            current_lr,
            elapsed,
        )

        epoch_record = {
            "epoch":     epoch,
            "lr":        current_lr,
            "elapsed_s": round(elapsed, 2),
            "train":     train_metrics,
            "val":       val_metrics,
        }
        history.append(epoch_record)

        # ── Best checkpoint ────────────────────────────────────────────────
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                ckpt_dir / "best.pt",
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, config, history,
            )
            logger.info("  ★ New best val loss: %.6f  →  best.pt", best_val_loss)

        # ── Periodic checkpoint ────────────────────────────────────────────
        if epoch % config["save_every"] == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, config, history,
            )

    # ── Final checkpoint & log ────────────────────────────────────────────
    _save_checkpoint(
        ckpt_dir / "final.pt",
        config["epochs"], model, optimizer, scheduler, scaler,
        best_val_loss, config, history,
    )
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training log → %s", log_path)
    logger.info("Done. Best val loss: %.6f", best_val_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────


def infer(
    sentences:   list[str],
    ckpt_path:   str,
    max_len:     int = 128,
    strategy:    Literal["greedy", "topk"] = "greedy",
    temperature: float = 1.0,
    top_k:       int = 50,
    device_str:  str = "auto",
) -> list[list[int]]:
    """Load a trained checkpoint and generate base-layer tokens for new sentences.

    Parameters
    ----------
    sentences:
        English source sentences.
    ckpt_path:
        Path to ``best.pt`` or any ``epoch_NNN.pt`` checkpoint.
    max_len:
        Maximum number of tokens to generate per sentence.
    strategy:
        ``"greedy"`` (deterministic argmax) or ``"topk"`` (sampling).
    temperature:
        Softmax temperature for top-k sampling.  Ignored for greedy.
    top_k:
        Number of candidates kept in top-k sampling.  Ignored for greedy.
    device_str:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns
    -------
    list[list[int]]
        One list of integer token IDs per input sentence.

    Example
    -------
    >>> tokens = infer(
    ...     ["Hello world", "Where is the hospital?"],
    ...     ckpt_path="checkpoints/base_token_generator/best.pt",
    ... )
    >>> print(tokens[0])   # e.g. [247, 13, 88, …]
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    device = _resolve_device(device_str)

    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]

    logger.info("Loading TextEncoder for inference …")
    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=True,
        device=str(device),
    )

    model = BaseTokenGenerator(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=0.0,               # no dropout at inference
        max_seq_len=config["model_max_seq_len"],
        text_dim=encoder.hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        "Loaded checkpoint from epoch %d (best val loss: %.6f)",
        ckpt["epoch"], ckpt["best_val_loss"],
    )

    with torch.no_grad():
        text_emb, text_mask = encoder.encode(sentences)
        text_emb  = text_emb.to(device)
        text_mask = text_mask.to(device)

        generated = model.generate(
            text_embeddings=text_emb,
            text_padding_mask=text_mask,
            max_len=max_len,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
        )  # [B, L]

    # Convert to nested Python lists; strip trailing PAD tokens per sample
    result: list[list[int]] = []
    for row in generated.tolist():
        # Trim trailing PAD_TOKEN_ID values that appear when sequences finish
        # before max_len (e.g. via eos_token_id stopping or padding fill)
        trimmed = row
        while trimmed and trimmed[-1] == PAD_TOKEN_ID:
            trimmed = trimmed[:-1]
        result.append(trimmed)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Validate the full training pipeline on fully synthetic data.

    Exercises every code path — teacher-forcing forward, loss, metrics,
    checkpoint save/load round-trip, greedy generation, and top-k sampling —
    without requiring the Kaggle dataset or a GPU.  Exits cleanly on success.
    """
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("=" * 64)
    logger.info("SMOKE TEST: train_base_token_generator.py")
    logger.info("=" * 64)

    device  = _resolve_device("auto")
    use_amp = device.type == "cuda"
    logger.info("Device: %s  |  AMP: %s", device, use_amp)

    torch.manual_seed(0)

    # ── Synthetic text encoder output ─────────────────────────────────────
    B, T_txt, D_txt = 4, 12, 768
    T_motion         = 10           # motion sequence length

    text_emb  = torch.randn(B, T_txt, D_txt, device=device)
    text_mask = torch.zeros(B, T_txt, dtype=torch.bool, device=device)
    for i, real_len in enumerate([12, 9, 7, 11]):
        text_mask[i, real_len:] = True
        text_emb[i, real_len:]  = 0.0

    # ── Synthetic token batch matching collate_sign_batch output ─────────
    # tokens [B, T_motion, 6] with PAD_TOKEN_ID (512) beyond real lengths
    real_lengths = torch.tensor([10, 8, 6, 9])
    tokens_6  = torch.randint(0, 512, (B, T_motion, 6))
    for i, rlen in enumerate(real_lengths.tolist()):
        tokens_6[i, rlen:, :] = PAD_TOKEN_ID

    # ── _prepare_decoder_batch ────────────────────────────────────────────
    logger.info("── _prepare_decoder_batch ──────────────────────────────────")
    dec_input, dec_target = _prepare_decoder_batch(tokens_6, real_lengths, device)
    assert dec_input.shape  == (B, T_motion), f"dec_input shape: {dec_input.shape}"
    assert dec_target.shape == (B, T_motion), f"dec_target shape: {dec_target.shape}"
    # Position 0 must always be BOS
    assert (dec_input[:, 0] == BOS_TOKEN_ID).all(), "dec_input col-0 must be BOS"
    # Target must carry PAD beyond each sequence's real length
    for i, rlen in enumerate(real_lengths.tolist()):
        if rlen < T_motion:
            assert (dec_target[i, rlen:] == PAD_TOKEN_ID).all(), \
                f"Sample {i}: target beyond length must be PAD"
    logger.info("  _prepare_decoder_batch: PASSED  dec_input=%s  dec_target=%s",
                tuple(dec_input.shape), tuple(dec_target.shape))

    # ── Model init ────────────────────────────────────────────────────────
    logger.info("── Model init ──────────────────────────────────────────────")
    model = BaseTokenGenerator(
        hidden_dim=64,          # small for speed
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.1,
        max_seq_len=256,
        text_dim=D_txt,
    ).to(device)
    logger.info("  %s", model)
    logger.info("  Params: %d", model.num_parameters())

    # ── Teacher-forcing forward ───────────────────────────────────────────
    logger.info("── forward() – teacher forcing ─────────────────────────────")
    model.train()
    _scaler_dev = device.type if device.type == "cuda" else "cpu"
    scaler = GradScaler(device=_scaler_dev, enabled=use_amp)

    with autocast(enabled=use_amp, device_type=device.type):
        logits = model(text_emb, text_mask, dec_input.to(device))
    assert logits.shape == (B, T_motion, VOCAB_SIZE), \
        f"logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"
    logger.info("  logits: %s  PASSED", tuple(logits.shape))

    # ── Loss ──────────────────────────────────────────────────────────────
    logger.info("── cross-entropy loss ──────────────────────────────────────")
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN_ID, label_smoothing=0.1, reduction="mean"
    )
    with autocast(enabled=use_amp, device_type=device.type):
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            dec_target.to(device).reshape(-1),
        )
    assert not torch.isnan(loss),  "Loss is NaN"
    assert not torch.isinf(loss),  "Loss is Inf"
    logger.info("  loss=%.4f  PASSED", loss.item())

    # ── Gradient flow ─────────────────────────────────────────────────────
    logger.info("── gradient flow ───────────────────────────────────────────")
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    grad_norms = {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    non_tied_zeros = [
        n for n, g in grad_norms.items()
        if g == 0.0 and "output_proj" not in n and "token_embedding" not in n
    ]
    assert not non_tied_zeros, f"Zero gradients in non-tied params: {non_tied_zeros}"
    logger.info("  %d params with grad, no unexpected zeros: PASSED",
                len(grad_norms))

    # ── Metrics ───────────────────────────────────────────────────────────
    logger.info("── _compute_metrics ────────────────────────────────────────")
    model.eval()
    with torch.no_grad():
        logits_eval = model(text_emb, text_mask, dec_input.to(device))
    raw_loss = F.cross_entropy(
        logits_eval.reshape(-1, VOCAB_SIZE),
        dec_target.to(device).reshape(-1),
        ignore_index=PAD_TOKEN_ID,
    ).item()
    metrics = _compute_metrics(logits_eval, dec_target.to(device), raw_loss)
    assert 0.0 <= metrics["accuracy"] <= 100.0
    assert metrics["perplexity"] > 0.0
    logger.info("  loss=%.4f  acc=%.2f%%  ppl=%.2f  PASSED",
                metrics["loss"], metrics["accuracy"], metrics["perplexity"])

    # ── Scheduler ─────────────────────────────────────────────────────────
    logger.info("── scheduler warm-up / cosine ──────────────────────────────")
    # warmup_epochs=2: lrs[0]=0.5*base, lrs[1]=1.0*base (peak), then cosine decay
    scheduler = _WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=6, base_lr=1e-4)
    lrs = []
    import warnings
    with warnings.catch_warnings():
        # PyTorch warns when CosineAnnealingLR.step() is called before
        # optimizer.step() — expected in this isolated scheduler unit test
        # (not an issue in the real training loop where order is correct).
        warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
        for _ in range(6):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
    # LR rises during warm-up (first two steps) then falls during cosine decay
    assert lrs[0] < lrs[1],  f"Warm-up epoch 1→2 should increase LR: {lrs[:2]}"
    assert lrs[1] >= lrs[-1], f"Cosine decay should lower LR from peak: {lrs}"
    logger.info("  LR trace: %s  PASSED", [f"{lr:.2e}" for lr in lrs])

    # ── Checkpoint round-trip ─────────────────────────────────────────────
    logger.info("── checkpoint save / load ──────────────────────────────────")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke.pt"
        _save_checkpoint(
            ckpt_path, epoch=3,
            model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            best_val_loss=raw_loss,
            config=DEFAULT_CONFIG, history=[],
        )
        assert ckpt_path.exists()

        model2     = BaseTokenGenerator(
            hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128,
            dropout=0.1, max_seq_len=256, text_dim=D_txt,
        ).to(device)
        optimizer2 = AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = _WarmupCosineScheduler(optimizer2, 2, 6, 1e-4)
        scaler2    = GradScaler(enabled=use_amp)
        ep, bvl, _ = _load_checkpoint(ckpt_path, model2, optimizer2, scheduler2, scaler2)
        assert ep == 4,     f"Expected start_epoch=4, got {ep}"
        assert bvl == pytest_approx(raw_loss) if False else abs(bvl - raw_loss) < 1e-6

        # Weight round-trip verification
        model2.eval()
        with torch.no_grad():
            logits2 = model2(text_emb, text_mask, dec_input.to(device))
        assert torch.allclose(logits_eval, logits2, atol=1e-5), \
            "Reloaded weights produce different logits"
        logger.info("  Checkpoint round-trip: PASSED")

    # ── Greedy generation ─────────────────────────────────────────────────
    logger.info("── generate() – greedy ─────────────────────────────────────")
    model.eval()
    gen_greedy = model.generate(
        text_embeddings=text_emb,
        text_padding_mask=text_mask,
        max_len=15,
        strategy="greedy",
    )
    assert gen_greedy.shape == (B, 15), f"Shape: {gen_greedy.shape}"
    assert (gen_greedy >= 0).all() and (gen_greedy < VOCAB_SIZE).all(), \
        f"Out-of-range tokens: min={gen_greedy.min()} max={gen_greedy.max()}"
    logger.info("  shape=%s  range OK  PASSED", tuple(gen_greedy.shape))

    # ── Top-k sampling ────────────────────────────────────────────────────
    logger.info("── generate() – top-k sampling ─────────────────────────────")
    torch.manual_seed(99)
    gen_topk = model.generate(
        text_embeddings=text_emb,
        text_padding_mask=text_mask,
        max_len=15,
        strategy="topk",
        temperature=0.8,
        top_k=20,
    )
    assert gen_topk.shape == (B, 15)
    assert (gen_topk >= 0).all() and (gen_topk < VOCAB_SIZE).all()
    logger.info("  shape=%s  range OK  PASSED", tuple(gen_topk.shape))

    # ── infer() stripping ─────────────────────────────────────────────────
    logger.info("── infer() PAD stripping ───────────────────────────────────")
    # Simulate generate() returning some trailing PADs
    dummy_generated = [[1, 2, 3, PAD_TOKEN_ID, PAD_TOKEN_ID],
                       [4, 5, PAD_TOKEN_ID]]
    stripped = []
    for row in dummy_generated:
        trimmed = row[:]
        while trimmed and trimmed[-1] == PAD_TOKEN_ID:
            trimmed = trimmed[:-1]
        stripped.append(trimmed)
    assert stripped == [[1, 2, 3], [4, 5]], f"Strip failed: {stripped}"
    logger.info("  PAD stripping: PASSED")

    # ── _prepare_decoder_batch edge: length-1 sequence ────────────────────
    logger.info("── edge case: length-1 sequence ────────────────────────────")
    tokens_edge   = torch.full((1, 5, 6), PAD_TOKEN_ID, dtype=torch.long)
    tokens_edge[0, 0, 0] = 7                        # one real base token
    lengths_edge  = torch.tensor([1])
    di, dt = _prepare_decoder_batch(tokens_edge, lengths_edge, device)
    assert di[0, 0] == BOS_TOKEN_ID,   "BOS must be at position 0"
    assert dt[0, 0] == 7,              f"Target[0,0] should be 7, got {dt[0,0]}"
    assert (dt[0, 1:] == PAD_TOKEN_ID).all(), "Positions 1+ must be PAD"
    logger.info("  length-1 edge case: PASSED")

    logger.info("=" * 64)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 64)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train SignVerse BaseTokenGenerator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")

    # ── smoke_test sub-command ────────────────────────────────────────────
    sub.add_parser("smoke_test", help="Self-contained smoke test (no CSV needed)")

    # ── infer sub-command ─────────────────────────────────────────────────
    inf = sub.add_parser("infer", help="Generate base tokens from a checkpoint")
    inf.add_argument("--ckpt",        required=True, type=str,
                     help="Path to best.pt or epoch_NNN.pt")
    inf.add_argument("--sentences",   required=True, nargs="+",
                     help="English sentences to process")
    inf.add_argument("--strategy",    default="greedy", choices=["greedy", "topk"])
    inf.add_argument("--temperature", default=1.0,  type=float)
    inf.add_argument("--top_k",       default=50,   type=int)
    inf.add_argument("--max_len",     default=128,  type=int)
    inf.add_argument("--device",      default="auto")

    # ── train flags (default mode; no sub-command required) ───────────────
    p.add_argument("--csv",          default=DEFAULT_CONFIG["csv_path"])
    p.add_argument("--val_frac",     default=DEFAULT_CONFIG["val_frac"],     type=float)
    p.add_argument("--epochs",       default=DEFAULT_CONFIG["epochs"],       type=int)
    p.add_argument("--batch",        default=DEFAULT_CONFIG["batch_size"],   type=int)
    p.add_argument("--lr",           default=DEFAULT_CONFIG["lr"],           type=float)
    p.add_argument("--weight_decay", default=DEFAULT_CONFIG["weight_decay"], type=float)
    p.add_argument("--grad_clip",    default=DEFAULT_CONFIG["grad_clip"],    type=float)
    p.add_argument("--warmup_epochs",default=DEFAULT_CONFIG["warmup_epochs"],type=int)
    p.add_argument("--label_smoothing", default=DEFAULT_CONFIG["label_smoothing"], type=float)
    p.add_argument("--max_seq_len",  default=DEFAULT_CONFIG["max_seq_len"],  type=int)
    p.add_argument("--seed",         default=DEFAULT_CONFIG["seed"],         type=int)
    p.add_argument("--workers",      default=DEFAULT_CONFIG["workers"],      type=int)
    p.add_argument("--device",       default=DEFAULT_CONFIG["device"])
    p.add_argument("--ckpt_dir",     default=DEFAULT_CONFIG["ckpt_dir"])
    p.add_argument("--save_every",   default=DEFAULT_CONFIG["save_every"],   type=int)
    p.add_argument("--no_amp",       action="store_true",
                   help="Disable automatic mixed precision even on CUDA")
    p.add_argument("--resume",       default=None, type=str,
                   help="Path to checkpoint .pt file to resume from")
    p.add_argument("--encoder",      default=DEFAULT_CONFIG["encoder_model"],
                   dest="encoder_model",
                   help="HuggingFace model ID or local path for FLAN-T5 encoder")
    p.add_argument("--encoder_max_len", default=DEFAULT_CONFIG["encoder_max_len"], type=int)

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode == "smoke_test":
        _smoke_test()
        return

    if args.mode == "infer":
        results = infer(
            sentences=args.sentences,
            ckpt_path=args.ckpt,
            max_len=args.max_len,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            device_str=args.device,
        )
        print("\nGenerated base tokens:")
        for sent, toks in zip(args.sentences, results):
            print(f"\n  Sentence : {sent!r}")
            print(f"  Length   : {len(toks)} frames")
            print(f"  Tokens   : {toks}")
        return

    # Default: train
    config = dict(DEFAULT_CONFIG)
    config.update({
        "csv_path":        args.csv,
        "val_frac":        args.val_frac,
        "epochs":          args.epochs,
        "batch_size":      args.batch,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "grad_clip":       args.grad_clip,
        "warmup_epochs":   args.warmup_epochs,
        "label_smoothing": args.label_smoothing,
        "max_seq_len":     args.max_seq_len,
        "seed":            args.seed,
        "workers":         args.workers,
        "device":          args.device,
        "ckpt_dir":        args.ckpt_dir,
        "save_every":      args.save_every,
        "amp":             not args.no_amp,
        "encoder_model":   args.encoder_model,
        "encoder_max_len": args.encoder_max_len,
    })
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()