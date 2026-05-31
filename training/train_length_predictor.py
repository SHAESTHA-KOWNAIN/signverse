"""
training/train_length_predictor.py
SignVerse – Training script for LengthPredictor.

Trains the LengthPredictor (models/length_predictor.py) to regress motion
sequence length from FLAN-T5 encoder embeddings (models/text_encoder.py),
using the SignVerse dataset (datasets/sign_dataset.py).

Pipeline
--------
    train.csv
        ↓  SignDataset (80 / 20 split, reproducible seed)
        ↓  DataLoader  (padded batches)
        ↓  TextEncoder (frozen FLAN-T5-base, fp32)
        ↓  LengthPredictor (MLP regression head, trainable)
        ↓  Huber loss  (normalised targets)
        ↓  AdamW + cosine-annealing LR schedule
        ↓  checkpoint every N epochs + best-val-loss checkpoint

Usage
-----
    # Minimal (defaults):
    python training/train_length_predictor.py --csv path/to/train.csv

    # Full control:
    python training/train_length_predictor.py \\
        --csv       data/train.csv         \\
        --ckpt_dir  checkpoints/length     \\
        --epochs    30                     \\
        --batch     32                     \\
        --lr        3e-4                   \\
        --val_frac  0.2                    \\
        --seed      42                     \\
        --workers   4                      \\
        --device    cuda

    # Resume from a checkpoint:
    python training/train_length_predictor.py \\
        --csv  data/train.csv \\
        --resume checkpoints/length/best.pt

    # Run smoke test (synthetic data, no CSV needed):
    python training/train_length_predictor.py --smoke_test

Checkpoint format
-----------------
    {
        "epoch":          int,
        "model_state":    OrderedDict,        # LengthPredictor weights
        "optimizer_state": dict,
        "scheduler_state": dict,
        "best_val_loss":  float,
        "config":         dict,               # all hyperparameters
        "train_history":  list[dict],         # per-epoch metrics
    }

Output
------
    checkpoints/
        best.pt          ← lowest validation loss
        epoch_{N:03d}.pt ← periodic snapshots (every --save_every epochs)
        training_log.json ← full metrics history
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

# ── Project imports (assumes script is run from the repo root) ─────────────────
# Adjust sys.path so imports work regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.sign_dataset import SignDataset, build_dataloader, collate_sign_batch
from models.text_encoder import TextEncoder
from models.length_predictor import LengthPredictor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass (plain dict for JSON-serializability)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # Data
    "csv_path":         "train.csv",
    "val_frac":         0.20,          # fraction of data held out for validation
    "max_seq_len":      256,           # SignDataset truncation (motion tokens)
    "seed":             42,
    # Training
    "epochs":           30,
    "batch_size":       32,
    "lr":               3e-4,
    "weight_decay":     1e-2,
    "huber_delta":      0.5,           # Huber loss δ parameter
    "grad_clip":        1.0,           # max gradient norm (0 = disabled)
    "warmup_epochs":    3,             # linear LR warm-up before cosine decay
    # Model
    "length_scale":     100.0,         # normalisation constant for targets
    "dropout":          0.1,
    "hidden_dims":      [256, 64],
    # Encoder
    "encoder_model":    "google/flan-t5-base",
    "encoder_max_len":  128,
    "encoder_freeze":   True,          # always True unless you have a huge GPU
    # Infrastructure
    "device":           "auto",
    "workers":          4,
    "ckpt_dir":         "checkpoints/length_predictor",
    "save_every":       5,             # save periodic checkpoint every N epochs
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _split_dataset(
    dataset: SignDataset,
    val_frac: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Reproducible 80/20 random split."""
    n_total = len(dataset)
    n_val   = max(1, math.floor(n_total * val_frac))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(dataset, [n_train, n_val], generator=generator)
    logger.info("Split: train=%d  val=%d  (val_frac=%.2f)", n_train, n_val, val_frac)
    return train_sub, val_sub


def _build_loaders(
    train_sub: Subset,
    val_sub:   Subset,
    batch_size: int,
    workers:    int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap subsets in DataLoaders with the SignVerse collate function."""
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_sign_batch,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sign_batch,
        pin_memory=pin,
        drop_last=False,
    )
    return train_loader, val_loader


def _compute_metrics(
    preds:   torch.Tensor,   # [B] raw frame predictions
    targets: torch.Tensor,   # [B] raw frame targets
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a batch of length predictions.

    All metrics are in raw frame units (de-normalised), making them directly
    interpretable as frame-count errors.
    """
    with torch.no_grad():
        abs_err = (preds - targets).abs()
        mae     = abs_err.mean().item()
        rmse    = ((preds - targets).pow(2).mean().sqrt()).item()
        # MAPE: clamp denominator to avoid div-by-zero on zero-length targets
        mape    = (abs_err / targets.clamp(min=1.0)).mean().item() * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ──────────────────────────────────────────────────────────────────────────────
# Warm-up scheduler wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _WarmupCosineScheduler:
    """Linear warm-up for the first `warmup_epochs`, then CosineAnnealing."""

    def __init__(
        self,
        optimizer:      torch.optim.Optimizer,
        warmup_epochs:  int,
        total_epochs:   int,
        base_lr:        float,
        eta_min:        float = 1e-6,
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
            # Linear warm-up: scale LR from 0 → base_lr
            frac = self._epoch / max(1, self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * frac
        else:
            self.cosine.step()

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "_epoch":  self._epoch,
            "cosine":  self.cosine.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["_epoch"]
        self.cosine.load_state_dict(state["cosine"])


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval epoch functions
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    encoder:    TextEncoder,
    predictor:  LengthPredictor,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    length_scale: float,
    huber_delta:  float,
    grad_clip:    float,
) -> dict[str, float]:
    """Run one full training epoch.

    Returns
    -------
    dict with keys: loss, mae, rmse, mape  (all averages over the epoch)
    """
    predictor.train()
    encoder.encoder.eval()   # encoder always stays in eval mode (frozen)

    total_loss = 0.0
    all_preds:   list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    n_batches = 0

    for batch in loader:
        sentences:  list[str]     = batch["sentences"]
        lengths:    torch.Tensor  = batch["lengths"].float().to(device)

        # ── 1. Encode text (no grad; encoder is frozen) ────────────────────
        with torch.no_grad():
            embeddings, padding_mask = encoder.encode(sentences)
        # Move to predictor device in case encoder lives on a different device
        embeddings   = embeddings.to(device)
        padding_mask = padding_mask.to(device)

        # ── 2. Predict lengths ────────────────────────────────────────────
        predicted = predictor(embeddings, padding_mask)   # [B]

        # ── 3. Normalised Huber loss ──────────────────────────────────────
        pred_norm   = predicted / length_scale
        target_norm = lengths   / length_scale
        loss = F.huber_loss(pred_norm, target_norm, delta=huber_delta)

        # ── 4. Backprop ───────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(predicted.detach())
        all_targets.append(lengths.detach())
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def _val_epoch(
    encoder:    TextEncoder,
    predictor:  LengthPredictor,
    loader:     DataLoader,
    device:     torch.device,
    length_scale: float,
    huber_delta:  float,
) -> dict[str, float]:
    """Run one full validation epoch (no gradient computation)."""
    predictor.eval()
    encoder.encoder.eval()

    total_loss = 0.0
    all_preds:   list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    n_batches = 0

    for batch in loader:
        sentences: list[str]    = batch["sentences"]
        lengths:   torch.Tensor = batch["lengths"].float().to(device)

        embeddings, padding_mask = encoder.encode(sentences)
        embeddings   = embeddings.to(device)
        padding_mask = padding_mask.to(device)

        predicted = predictor(embeddings, padding_mask)

        pred_norm   = predicted / length_scale
        target_norm = lengths   / length_scale
        loss = F.huber_loss(pred_norm, target_norm, delta=huber_delta)

        total_loss += loss.item()
        all_preds.append(predicted)
        all_targets.append(lengths)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    path:      Path,
    epoch:     int,
    predictor: LengthPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: _WarmupCosineScheduler,
    best_val_loss: float,
    config:    dict,
    history:   list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":            epoch,
            "model_state":      predictor.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "best_val_loss":    best_val_loss,
            "config":           config,
            "train_history":    history,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def _load_checkpoint(
    path:      Path,
    predictor: LengthPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: _WarmupCosineScheduler,
) -> tuple[int, float, list[dict]]:
    """Load checkpoint in-place.  Returns (start_epoch, best_val_loss, history)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    history       = ckpt.get("train_history", [])
    logger.info(
        "Resumed from %s (epoch=%d, best_val_loss=%.6f)",
        path, ckpt["epoch"], best_val_loss,
    )
    return start_epoch, best_val_loss, history


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume: Optional[str] = None) -> None:
    """Full training loop.

    Parameters
    ----------
    config:
        Hyperparameter dict (see DEFAULT_CONFIG).
    resume:
        Path to a checkpoint .pt file to resume from, or None.
    """
    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = _resolve_device(config["device"])
    logger.info("Device: %s", device)

    # ── Dataset & split ──────────────────────────────────────────────────────
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
        "Train batches/epoch: %d  |  Val batches/epoch: %d",
        len(train_loader), len(val_loader),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    logger.info("Loading TextEncoder …")
    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=config["encoder_freeze"],
        device=str(device),
    )
    logger.info("  %s", encoder)

    predictor = LengthPredictor(
        input_dim=encoder.hidden_dim,
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
        length_scale=config["length_scale"],
    ).to(device)
    logger.info("  %s", predictor)
    logger.info(
        "  Trainable parameters: %d", predictor.num_parameters(trainable_only=True)
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        predictor.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = _WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["epochs"],
        base_lr=config["lr"],
    )

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = math.inf
    history:      list[dict] = []

    if resume:
        start_epoch, best_val_loss, history = _load_checkpoint(
            Path(resume), predictor, optimizer, scheduler
        )

    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info(
        "Starting training: epochs=%d  start_epoch=%d  batch=%d  lr=%.2e",
        config["epochs"], start_epoch, config["batch_size"], config["lr"],
    )

    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.time()

        train_metrics = _train_epoch(
            encoder=encoder,
            predictor=predictor,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            length_scale=config["length_scale"],
            huber_delta=config["huber_delta"],
            grad_clip=config["grad_clip"],
        )
        val_metrics = _val_epoch(
            encoder=encoder,
            predictor=predictor,
            loader=val_loader,
            device=device,
            length_scale=config["length_scale"],
            huber_delta=config["huber_delta"],
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        # ── Log ──────────────────────────────────────────────────────────
        logger.info(
            "Epoch %3d/%d | "
            "train loss=%.5f mae=%.1f rmse=%.1f mape=%.1f%% | "
            "val   loss=%.5f mae=%.1f rmse=%.1f mape=%.1f%% | "
            "lr=%.2e | %.1fs",
            epoch, config["epochs"],
            train_metrics["loss"],
            train_metrics["mae"], train_metrics["rmse"], train_metrics["mape"],
            val_metrics["loss"],
            val_metrics["mae"], val_metrics["rmse"], val_metrics["mape"],
            current_lr,
            elapsed,
        )

        epoch_record = {
            "epoch":  epoch,
            "lr":     current_lr,
            "elapsed_s": round(elapsed, 2),
            "train":  train_metrics,
            "val":    val_metrics,
        }
        history.append(epoch_record)

        # ── Best checkpoint ───────────────────────────────────────────────
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                ckpt_dir / "best.pt",
                epoch, predictor, optimizer, scheduler,
                best_val_loss, config, history,
            )
            logger.info(
                "  ★ New best val loss: %.6f  →  saved best.pt", best_val_loss
            )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if epoch % config["save_every"] == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                epoch, predictor, optimizer, scheduler,
                best_val_loss, config, history,
            )

    # ── Save final checkpoint & training log ─────────────────────────────────
    _save_checkpoint(
        ckpt_dir / "final.pt",
        config["epochs"], predictor, optimizer, scheduler,
        best_val_loss, config, history,
    )

    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training log saved → %s", log_path)

    logger.info(
        "Training complete. Best val loss: %.6f", best_val_loss
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference example
# ──────────────────────────────────────────────────────────────────────────────

def infer(
    sentences:  list[str],
    ckpt_path:  str,
    device_str: str = "auto",
) -> list[int]:
    """Load a trained checkpoint and predict motion lengths for new sentences.

    Parameters
    ----------
    sentences:
        English sentences to predict lengths for.
    ckpt_path:
        Path to a saved checkpoint (``best.pt`` or ``epoch_NNN.pt``).
    device_str:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns
    -------
    list[int]
        Predicted discrete frame counts, one per sentence.

    Example
    -------
    >>> lengths = infer(
    ...     ["Hello, how are you?", "Where is the nearest hospital?"],
    ...     ckpt_path="checkpoints/length_predictor/best.pt",
    ... )
    >>> print(lengths)   # e.g. [72, 95]
    """
    device = _resolve_device(device_str)
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]

    logger.info("Loading encoder …")
    encoder = TextEncoder(
        model_name_or_path=config["encoder_model"],
        max_length=config["encoder_max_len"],
        freeze=True,
        device=str(device),
    )

    predictor = LengthPredictor(
        input_dim=encoder.hidden_dim,
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=0.0,         # no dropout at inference
        length_scale=config["length_scale"],
    ).to(device)
    predictor.load_state_dict(ckpt["model_state"])
    predictor.eval()

    logger.info(
        "Loaded checkpoint from epoch %d (best val loss: %.6f)",
        ckpt["epoch"], ckpt["best_val_loss"],
    )

    with torch.no_grad():
        embeddings, padding_mask = encoder.encode(sentences)
        embeddings   = embeddings.to(device)
        padding_mask = padding_mask.to(device)
        lengths = predictor.predict_lengths(embeddings, padding_mask)

    return lengths.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test (no CSV required)
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Validate the full training pipeline on synthetic data.

    Builds a tiny in-memory SignDataset lookalike using raw Python dicts,
    verifies that the train/val loop, checkpointing, and inference all run
    without error.  Exits with code 0 on success, 1 on failure.
    """
    import tempfile, os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("=" * 60)
    logger.info("SMOKE TEST: train_length_predictor.py")
    logger.info("=" * 60)

    device = _resolve_device("auto")
    logger.info("Device: %s", device)

    # ── Synthetic dataset using only the TextEncoder + LengthPredictor ─────
    # Bypass SignDataset / CSV; we build tensors directly so the test
    # runs without the Kaggle dataset present.

    B, T_text, D, T_motion = 6, 12, 768, 1
    torch.manual_seed(0)

    # Synthetic encoder outputs
    embeddings_all   = torch.randn(B, T_text, D)
    padding_mask_all = torch.zeros(B, T_text, dtype=torch.bool)
    # Vary sentence lengths to simulate realistic padding
    for i, real_len in enumerate([12, 9, 7, 11, 5, 10]):
        if real_len < T_text:
            padding_mask_all[i, real_len:] = True
            embeddings_all[i, real_len:]   = 0.0
    target_lengths = torch.tensor([98.0, 120.0, 75.0, 110.0, 60.0, 88.0])

    # ── Model init ────────────────────────────────────────────────────────
    logger.info("Initialising LengthPredictor …")
    predictor = LengthPredictor(
        input_dim=D,
        hidden_dims=(256, 64),
        dropout=0.1,
        length_scale=100.0,
    ).to(device)
    logger.info("  %s", predictor)

    optimizer = AdamW(predictor.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = _WarmupCosineScheduler(
        optimizer,
        warmup_epochs=1,
        total_epochs=4,
        base_lr=3e-4,
    )

    # ── Mini training loop (4 epochs) ─────────────────────────────────────
    logger.info("Running 4 synthetic training epochs …")
    loss_scale = 100.0
    for epoch in range(1, 5):
        predictor.train()
        emb  = embeddings_all.to(device)
        pmsk = padding_mask_all.to(device)
        tgt  = target_lengths.to(device)

        optimizer.zero_grad()
        preds = predictor(emb, pmsk)
        loss  = F.huber_loss(preds / loss_scale, tgt / loss_scale, delta=0.5)
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        logger.info(
            "  Epoch %d | loss=%.6f | lr=%.2e | preds=%s",
            epoch, loss.item(), scheduler.get_last_lr()[0],
            preds.detach().round().long().tolist(),
        )

    # ── Metrics ──────────────────────────────────────────────────────────
    predictor.eval()
    with torch.no_grad():
        preds_eval = predictor(embeddings_all.to(device), padding_mask_all.to(device))
    metrics = _compute_metrics(preds_eval, target_lengths.to(device))
    logger.info("Eval metrics: mae=%.2f  rmse=%.2f  mape=%.1f%%",
                metrics["mae"], metrics["rmse"], metrics["mape"])

    # ── Checkpoint save / load round-trip ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_test.pt"
        _save_checkpoint(
            ckpt_path,
            epoch=4,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=loss.item(),
            config=DEFAULT_CONFIG,
            history=[],
        )
        assert ckpt_path.exists(), "Checkpoint file not created"

        # Reload into a fresh predictor
        predictor2 = LengthPredictor(
            input_dim=D, hidden_dims=(256, 64), dropout=0.1, length_scale=100.0
        ).to(device)
        optimizer2  = AdamW(predictor2.parameters(), lr=3e-4)
        scheduler2  = _WarmupCosineScheduler(optimizer2, 1, 4, 3e-4)
        epoch_loaded, best_loss_loaded, _ = _load_checkpoint(
            ckpt_path, predictor2, optimizer2, scheduler2
        )
        assert epoch_loaded == 5, f"Expected start_epoch=5, got {epoch_loaded}"

        # Verify weight round-trip
        predictor2.eval()
        with torch.no_grad():
            preds2 = predictor2(embeddings_all.to(device), padding_mask_all.to(device))
        assert torch.allclose(preds_eval, preds2, atol=1e-4), \
            "Loaded weights differ from saved weights"
        logger.info("Checkpoint round-trip: PASSED")

    # ── predict_lengths discrete output ──────────────────────────────────
    discrete = predictor.predict_lengths(
        embeddings_all.to(device), padding_mask_all.to(device)
    )
    assert discrete.dim() == 1
    assert (discrete >= 1).all(), "predict_lengths must return values >= 1"
    logger.info("predict_lengths output: %s  (all >= 1: PASSED)", discrete.tolist())

    # ── _compute_metrics unit test ────────────────────────────────────────
    p = torch.tensor([100.0, 80.0, 120.0])
    t = torch.tensor([100.0, 100.0, 100.0])
    m = _compute_metrics(p, t)
    assert abs(m["mae"]  - (0 + 20 + 20) / 3) < 1e-3, f"MAE wrong: {m['mae']}"
    assert abs(m["rmse"] - math.sqrt((0 + 400 + 400) / 3)) < 1e-3, f"RMSE wrong: {m['rmse']}"
    logger.info("_compute_metrics unit test: PASSED (mae=%.2f rmse=%.2f mape=%.1f%%)",
                m["mae"], m["rmse"], m["mape"])

    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train or evaluate the SignVerse LengthPredictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")

    # ── train sub-command (also the default when --csv is given directly) ──
    # For convenience, all flags work without a sub-command:
    p.add_argument("--csv",        type=str,   default=DEFAULT_CONFIG["csv_path"],
                   help="Path to train.csv")
    p.add_argument("--val_frac",   type=float, default=DEFAULT_CONFIG["val_frac"])
    p.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--seed",       type=int,   default=DEFAULT_CONFIG["seed"])
    p.add_argument("--workers",    type=int,   default=DEFAULT_CONFIG["workers"])
    p.add_argument("--device",     type=str,   default=DEFAULT_CONFIG["device"])
    p.add_argument("--ckpt_dir",   type=str,   default=DEFAULT_CONFIG["ckpt_dir"])
    p.add_argument("--save_every", type=int,   default=DEFAULT_CONFIG["save_every"])
    p.add_argument("--max_seq_len",type=int,   default=DEFAULT_CONFIG["max_seq_len"])
    p.add_argument("--length_scale", type=float, default=DEFAULT_CONFIG["length_scale"])
    p.add_argument("--dropout",    type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--warmup_epochs", type=int, default=DEFAULT_CONFIG["warmup_epochs"])
    p.add_argument("--grad_clip",  type=float, default=DEFAULT_CONFIG["grad_clip"])
    p.add_argument("--huber_delta",type=float, default=DEFAULT_CONFIG["huber_delta"])
    p.add_argument("--encoder",    type=str,   default=DEFAULT_CONFIG["encoder_model"],
                   dest="encoder_model")
    p.add_argument("--encoder_max_len", type=int, default=DEFAULT_CONFIG["encoder_max_len"])

    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint .pt file to resume training from")

    # ── infer sub-command ─────────────────────────────────────────────────
    infer_p = sub.add_parser("infer", help="Run inference from a saved checkpoint")
    infer_p.add_argument("--ckpt",     type=str, required=True,
                         help="Path to best.pt or epoch_NNN.pt")
    infer_p.add_argument("--sentences", type=str, nargs="+", required=True,
                         help='English sentences (quote multi-word: "Hello world")')
    infer_p.add_argument("--device",   type=str, default="auto")

    # ── smoke_test sub-command ─────────────────────────────────────────────
    sub.add_parser("smoke_test", help="Run self-contained smoke test (no CSV needed)")

    # Legacy flag (backwards compat)
    p.add_argument("--smoke_test", action="store_true", help=argparse.SUPPRESS)

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Handle smoke_test (sub-command or legacy flag)
    if args.mode == "smoke_test" or getattr(args, "smoke_test", False):
        _smoke_test()
        return

    # Handle infer sub-command
    if args.mode == "infer":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        predicted = infer(
            sentences=args.sentences,
            ckpt_path=args.ckpt,
            device_str=args.device,
        )
        print("\nPredicted motion lengths (frames):")
        for sent, length in zip(args.sentences, predicted):
            print(f"  {length:>5}  ← {sent!r}")
        return

    # Default: train
    config = dict(DEFAULT_CONFIG)
    config.update({
        "csv_path":      args.csv,
        "val_frac":      args.val_frac,
        "epochs":        args.epochs,
        "batch_size":    args.batch,
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        "seed":          args.seed,
        "workers":       args.workers,
        "device":        args.device,
        "ckpt_dir":      args.ckpt_dir,
        "save_every":    args.save_every,
        "max_seq_len":   args.max_seq_len,
        "length_scale":  args.length_scale,
        "dropout":       args.dropout,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip":     args.grad_clip,
        "huber_delta":   args.huber_delta,
        "encoder_model": args.encoder_model,
        "encoder_max_len": args.encoder_max_len,
        "encoder_freeze": True,
    })
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()