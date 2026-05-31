"""
datasets/sign_dataset.py
SignVerse – PyTorch Dataset for English-to-Sign-Language motion tokens.

Each sample returns:
    {
        "sentence": str,                    # English source sentence
        "gloss":    str,                    # Normalised sign-language gloss
        "tokens":   LongTensor [T, 6],      # RVQ token matrix
    }

tokens[:, 0] = base_tokens
tokens[:, 1] = residual_1
tokens[:, 2] = residual_2
tokens[:, 3] = residual_3
tokens[:, 4] = residual_4
tokens[:, 5] = residual_5

Design decisions
----------------
* All 6 RVQ layers must have the same length; rows that violate this are
  skipped and logged (corrupt BVH / truncated export).
* Rows whose token strings are missing (NaN) are skipped.
* Anomalous rows where a single sign is repeated more than MAX_GLOSS_REPEAT
  times in the gloss are optionally skipped (enabled by default).
* Sequences longer than max_seq_len are truncated (not silently dropped) so
  the model never sees padding artefacts from extreme outliers.
* The collate function pads variable-length sequences in a batch to the
  length of the longest sequence in that batch (dynamic padding).
* A boolean mask tensor is returned so the model can ignore padding positions.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

LAYER_COLUMNS: list[str] = [
    "base_tokens",
    "residual_1",
    "residual_2",
    "residual_3",
    "residual_4",
    "residual_5",
]

CODEBOOK_SIZE: int = 512          # token IDs are in [0, 511]
PAD_TOKEN_ID:  int = CODEBOOK_SIZE  # 512 – out-of-range, never a real token
MAX_GLOSS_REPEAT: int = 10         # single sign repeated more than this → corrupt

# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_token_string(raw: str) -> list[int]:
    """Convert a space-separated integer string into a Python int list.

    Raises ValueError if any token is not a non-negative integer.
    """
    tokens = []
    for tok in raw.split():
        if not tok.isdigit():
            raise ValueError(f"Non-integer token encountered: {tok!r}")
        tokens.append(int(tok))
    return tokens


def _is_gloss_corrupt(gloss: str, threshold: int = MAX_GLOSS_REPEAT) -> bool:
    """Return True when a single sign is repeated more than *threshold* times.

    Catches annotation errors where a BVH loop was incorrectly transcribed,
    e.g. 'PLACE PLACE PLACE … PLACE' (49×) for 'She patted her hair into place.'
    """
    cleaned = gloss.rstrip("/").strip()
    words = cleaned.split()
    if not words:
        return True
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count > threshold


def _normalise_gloss(gloss: str) -> str:
    """Uppercase, strip trailing // terminator, collapse whitespace."""
    return " ".join(gloss.upper().rstrip("/").split())


# ── Dataset ────────────────────────────────────────────────────────────────────


class SignDataset(Dataset):
    """PyTorch Dataset for SignVerse RVQ motion token sequences.

    Parameters
    ----------
    csv_path:
        Path to train.csv (or any CSV with the expected schema).
    max_seq_len:
        Sequences longer than this are truncated to the first *max_seq_len*
        frames.  Set to None to disable truncation.
    skip_corrupt_gloss:
        When True (default), rows with anomalous looping glosses are skipped.
    """

    def __init__(
        self,
        csv_path: str | Path,
        max_seq_len: Optional[int] = 256,
        skip_corrupt_gloss: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.max_seq_len = max_seq_len
        self.skip_corrupt_gloss = skip_corrupt_gloss

        self.samples: list[dict] = []
        self._load()

    # ── Loading & validation ────────────────────────────────────────────────

    def _load(self) -> None:
        logger.info("Loading dataset from %s", self.csv_path)

        df = pd.read_csv(self.csv_path, dtype=str)

        required_cols = {"sentence", "gloss"} | set(LAYER_COLUMNS)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}\n"
                f"Found columns: {list(df.columns)}"
            )

        skipped_null    = 0
        skipped_corrupt = 0
        skipped_length  = 0
        skipped_range   = 0
        loaded          = 0

        for row_idx, row in df.iterrows():
            # ── 1. Skip rows with any missing token column ──────────────────
            if any(pd.isna(row[col]) for col in LAYER_COLUMNS):
                skipped_null += 1
                logger.debug("Row %d skipped: one or more token columns are NaN", row_idx)
                continue

            # ── 2. Optionally skip corrupt gloss ────────────────────────────
            gloss_raw = str(row["gloss"])
            if self.skip_corrupt_gloss and _is_gloss_corrupt(gloss_raw):
                skipped_corrupt += 1
                logger.debug(
                    "Row %d skipped: corrupt/looping gloss %r", row_idx, gloss_raw[:60]
                )
                continue

            # ── 3. Parse all 6 token columns ────────────────────────────────
            try:
                layers: list[list[int]] = [
                    _parse_token_string(str(row[col])) for col in LAYER_COLUMNS
                ]
            except ValueError as exc:
                skipped_corrupt += 1
                logger.debug("Row %d skipped: token parse error – %s", row_idx, exc)
                continue

            # ── 4. Validate equal length across all 6 layers ─────────────
            lengths = [len(layer) for layer in layers]
            if len(set(lengths)) != 1:
                skipped_length += 1
                logger.debug(
                    "Row %d skipped: layer length mismatch %s", row_idx, lengths
                )
                continue

            seq_len = lengths[0]

            # ── 5. Validate token ID range ───────────────────────────────
            invalid_ids = [
                tok
                for layer in layers
                for tok in layer
                if not (0 <= tok < CODEBOOK_SIZE)
            ]
            if invalid_ids:
                skipped_range += 1
                logger.debug(
                    "Row %d skipped: token IDs out of [0, %d): %s",
                    row_idx,
                    CODEBOOK_SIZE,
                    invalid_ids[:5],
                )
                continue

            # ── 6. Truncate extreme sequences ────────────────────────────
            if self.max_seq_len is not None and seq_len > self.max_seq_len:
                layers = [layer[: self.max_seq_len] for layer in layers]
                logger.debug(
                    "Row %d truncated from %d to %d tokens",
                    row_idx,
                    seq_len,
                    self.max_seq_len,
                )

            # ── 7. Build [T, 6] token tensor ────────────────────────────
            # Stack layers column-wise: shape [T, 6]
            token_tensor: Tensor = torch.tensor(
                list(zip(*layers)), dtype=torch.long
            )  # zip(*layers) transposes [[T]×6] → [T×[6]]

            self.samples.append(
                {
                    "sentence": str(row["sentence"]),
                    "gloss":    _normalise_gloss(gloss_raw),
                    "tokens":   token_tensor,
                }
            )
            loaded += 1

        total = len(df)
        logger.info(
            "Dataset loaded: %d/%d rows kept  |  skipped – null=%d  corrupt=%d  "
            "length_mismatch=%d  range_error=%d",
            loaded,
            total,
            skipped_null,
            skipped_corrupt,
            skipped_length,
            skipped_range,
        )

        if loaded == 0:
            raise RuntimeError(
                f"No valid samples found in {self.csv_path}. "
                "Check the file path and column schema."
            )

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample dict with keys: sentence, gloss, tokens."""
        return self.samples[idx]

    # ── Introspection helpers ──────────────────────────────────────────────

    def token_lengths(self) -> Tensor:
        """Return a 1-D LongTensor of per-sample sequence lengths."""
        return torch.tensor([s["tokens"].shape[0] for s in self.samples])

    def summary(self) -> str:
        """One-line summary string, useful for logging."""
        lengths = self.token_lengths().float()
        return (
            f"SignDataset | samples={len(self)} | "
            f"seq_len mean={lengths.mean():.1f} "
            f"min={lengths.min().item()} "
            f"max={lengths.max().item()} "
            f"p50={lengths.median().item():.0f}"
        )


# ── Collate & DataLoader ───────────────────────────────────────────────────────


def collate_sign_batch(batch: list[dict]) -> dict:
    """Collate a list of samples into a padded batch.

    Returns
    -------
    dict with keys:
        sentences  : list[str],                length B
        glosses    : list[str],                length B
        tokens     : LongTensor [B, T_max, 6], padded with PAD_TOKEN_ID
        lengths    : LongTensor [B],           original (unpadded) sequence lengths
        padding_mask : BoolTensor [B, T_max],  True where the position is padding
    """
    sentences = [s["sentence"] for s in batch]
    glosses   = [s["gloss"]    for s in batch]
    tokens    = [s["tokens"]   for s in batch]  # each is [T_i, 6]

    lengths   = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    t_max     = int(lengths.max().item())
    b         = len(batch)

    # Allocate padded tensor filled with PAD_TOKEN_ID
    padded = tokens[0].new_full((b, t_max, 6), fill_value=PAD_TOKEN_ID)
    for i, (tok, length) in enumerate(zip(tokens, lengths)):
        padded[i, : length.item(), :] = tok

    # padding_mask: True at positions that are padding (for use with nn.Transformer
    # key_padding_mask convention where True = ignore)
    padding_mask = torch.arange(t_max).unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T_max]

    return {
        "sentences":    sentences,
        "glosses":      glosses,
        "tokens":       padded,           # [B, T_max, 6]
        "lengths":      lengths,          # [B]
        "padding_mask": padding_mask,     # [B, T_max]
    }


def build_dataloader(
    csv_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: Optional[int] = 256,
    skip_corrupt_gloss: bool = True,
    **dataloader_kwargs,
) -> DataLoader:
    """Convenience factory: dataset + DataLoader with the custom collate function.

    Parameters
    ----------
    csv_path:
        Path to train.csv.
    batch_size:
        Number of samples per batch.
    shuffle:
        Shuffle samples each epoch (set False for val/test).
    num_workers:
        Subprocesses for data loading.  Set 0 for debugging or Windows.
    max_seq_len:
        Passed through to SignDataset.
    skip_corrupt_gloss:
        Passed through to SignDataset.
    **dataloader_kwargs:
        Any additional keyword arguments forwarded to DataLoader
        (e.g. pin_memory=True, prefetch_factor=2).
    """
    dataset = SignDataset(
        csv_path=csv_path,
        max_seq_len=max_seq_len,
        skip_corrupt_gloss=skip_corrupt_gloss,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sign_batch,
        **dataloader_kwargs,
    )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"

    dataset = SignDataset(csv_path=csv_path, max_seq_len=256)
    print(dataset.summary())
    print()

    # Inspect first sample
    sample = dataset[0]
    print("── Sample 0 ──────────────────────────────────")
    print(f"  sentence : {sample['sentence']}")
    print(f"  gloss    : {sample['gloss']}")
    print(f"  tokens   : shape={tuple(sample['tokens'].shape)}  dtype={sample['tokens'].dtype}")
    print(f"  tokens[0]: {sample['tokens'][0].tolist()}  (first frame, all 6 layers)")
    print(f"  tokens[-1]: {sample['tokens'][-1].tolist()}  (last frame, all 6 layers)")
    print()

    # Batch test
    loader = build_dataloader(
        csv_path=csv_path,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        max_seq_len=256,
    )
    batch = next(iter(loader))
    print("── Batch (size=8) ────────────────────────────")
    print(f"  tokens        : {tuple(batch['tokens'].shape)}")
    print(f"  lengths       : {batch['lengths'].tolist()}")
    print(f"  padding_mask  : {tuple(batch['padding_mask'].shape)}")
    pad_positions = batch["padding_mask"].sum().item()
    total_positions = batch["padding_mask"].numel()
    print(f"  padding ratio : {pad_positions}/{total_positions} "
          f"({100*pad_positions/total_positions:.1f}%)")
    print(f"  sentences[0]  : {batch['sentences'][0]}")
    print(f"  glosses[0]    : {batch['glosses'][0]}")