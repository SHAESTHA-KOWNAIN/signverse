"""
models/length_predictor.py
SignVerse – Sequence length predictor.

Predicts the number of motion frames (token sequence length) that corresponds
to a given English sentence, conditioned on FLAN-T5 encoder embeddings.

Architecture
------------
    1. Masked mean-pool  [B, T, 768] → [B, 768]
       Average over real (non-padding) token positions only.

    2. MLP regression head
       Linear(768 → 256) → GELU → Dropout
       Linear(256 →  64) → GELU → Dropout
       Linear( 64 →   1)
       Softplus activation  → guarantees a strictly positive prediction

    3. Output squeezed to [B]

Design notes
------------
* Softplus (β=1) is used instead of ReLU so the gradient never dies at zero
  and the output is smooth and always positive — important for a length that
  must be ≥ 1.
* The MLP hidden widths (256, 64) were chosen to be proportional to the
  input dimension (768) while keeping the predictor lightweight; it accounts
  for < 0.2 % of the full SignVerse parameter budget.
* Layer normalisation is applied to the pooled representation before the
  first linear layer.  This stabilises training when the upstream T5 encoder
  is frozen (its output distribution is fixed) and when it is fine-tuned
  (distribution shifts mid-training).
* Dropout is applied only during training; eval() disables it automatically.
* The module is compatible with both MSE and Huber (smooth L1) loss.

Training target
---------------
    Normalise raw frame counts by dividing by a reference scale (default 100,
    close to the dataset median of 98) so the regression target lives roughly
    in [0.25, 2.5] rather than [24, 256].  De-normalise at inference time via
    ``predicted_length * length_scale``.  Pass ``length_scale`` at construction
    time to bake it into the module.

Input / output contract
-----------------------
    forward(embeddings, padding_mask)
        embeddings   : FloatTensor [B, T, D]  — from TextEncoder.forward()
        padding_mask : BoolTensor  [B, T]     — True at padding positions
                                               (matches TextEncoder convention)

    Returns
        predicted_length : FloatTensor [B]    — positive predicted frame count
                                               in raw frames (de-normalised)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_INPUT_DIM:  int = 768   # flan-t5-base d_model
_HIDDEN_1:   int = 256
_HIDDEN_2:   int = 64
_DEFAULT_SCALE: float = 100.0   # dataset median ≈ 98; round to 100


# ── LengthPredictor ────────────────────────────────────────────────────────────


class LengthPredictor(nn.Module):
    """MLP that predicts motion sequence length from T5 encoder embeddings.

    Parameters
    ----------
    input_dim:
        Dimensionality of the incoming token embeddings.  Defaults to 768,
        which matches ``google/flan-t5-base``.
    hidden_dims:
        Sequence of hidden layer widths for the MLP.  Defaults to
        ``(256, 64)``.
    dropout:
        Dropout probability applied after each hidden activation.  Set to
        ``0.0`` to disable.
    length_scale:
        The model internally normalises targets by this value during training
        (caller's responsibility) and **multiplies** the raw MLP output by it
        before returning, so the final prediction is in raw frame units.
        Defaults to 100 (close to the SignVerse dataset median of 98 frames).
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        hidden_dims: tuple[int, ...] = (_HIDDEN_1, _HIDDEN_2),
        dropout: float = 0.1,
        length_scale: float = _DEFAULT_SCALE,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one element")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")

        self.input_dim    = input_dim
        self.hidden_dims  = hidden_dims
        self.dropout_p    = dropout
        self.length_scale = length_scale

        # ── Layer normalisation on pooled representation ──────────────────
        self.layer_norm = nn.LayerNorm(input_dim)

        # ── MLP ──────────────────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = out_dim

        # Final projection to scalar
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Softplus activation: smooth, always positive, gradient never zero
        self.output_activation = nn.Softplus(beta=1)

        self._init_weights()
        logger.info("LengthPredictor: %s", self)

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier uniform for linear weights; zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Masked mean pooling ────────────────────────────────────────────────

    @staticmethod
    def _masked_mean_pool(
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average token embeddings over real (non-padding) positions.

        Parameters
        ----------
        embeddings:
            ``FloatTensor[B, T, D]`` — token-level encoder output.
        padding_mask:
            ``BoolTensor[B, T]`` — ``True`` at padding positions.

        Returns
        -------
        pooled : ``FloatTensor[B, D]``
            Mean of real token embeddings per sample.  If a sample has no
            real tokens (degenerate all-padding row), its pooled vector is
            zeros (no NaN or Inf).
        """
        # real_mask: 1.0 at real tokens, 0.0 at padding  →  [B, T, 1]
        real_mask: torch.Tensor = (~padding_mask).to(embeddings.dtype).unsqueeze(-1)

        # Sum real token embeddings
        summed: torch.Tensor = (embeddings * real_mask).sum(dim=1)   # [B, D]

        # Count real tokens; clamp to 1 to prevent division by zero for
        # degenerate all-padding rows (returns zero vector, not NaN)
        counts: torch.Tensor = real_mask.sum(dim=1).clamp(min=1.0)   # [B, 1]

        return summed / counts  # [B, D]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict motion sequence length for a batch of sentences.

        Parameters
        ----------
        embeddings:
            ``FloatTensor[B, T, D]`` — contextualised token embeddings from
            ``TextEncoder.forward()``.  Padding positions should be
            zero-filled (TextEncoder does this automatically).
        padding_mask:
            ``BoolTensor[B, T]`` — ``True`` at positions that correspond to
            padding tokens.  Must be on the same device as ``embeddings``.

        Returns
        -------
        predicted_length : ``FloatTensor[B]``
            Predicted frame count per sample, expressed in raw (de-normalised)
            frame units.  All values are strictly positive.

        Notes
        -----
        * During training, compute your loss against
          ``target_frames / self.length_scale`` to match the internal scale.
          Example::

              target_normalised = target_frames.float() / predictor.length_scale
              loss = F.huber_loss(predictor(emb, mask) / predictor.length_scale,
                                  target_normalised)

          Or equivalently, divide both sides by ``length_scale`` before the
          loss::

              pred_norm   = predictor(emb, mask) / predictor.length_scale
              target_norm = target_frames.float() / predictor.length_scale
              loss = F.mse_loss(pred_norm, target_norm)

        * Inference: ``predicted_length`` is already in raw frame units —
          round to the nearest integer to obtain a discrete sequence length::

              seq_len = predicted_length.round().long().clamp(min=1)
        """
        # ── Shape / device validation ──────────────────────────────────
        if embeddings.dim() != 3:
            raise ValueError(
                f"embeddings must be 3-D [B, T, D], got shape {tuple(embeddings.shape)}"
            )
        if padding_mask.dim() != 2:
            raise ValueError(
                f"padding_mask must be 2-D [B, T], got shape {tuple(padding_mask.shape)}"
            )
        B, T, D = embeddings.shape
        if padding_mask.shape != (B, T):
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} does not match "
                f"embeddings batch/seq dims ({B}, {T})"
            )
        if D != self.input_dim:
            raise ValueError(
                f"embeddings last dim {D} != input_dim {self.input_dim}"
            )

        # ── 1. Masked mean pool  [B, T, D] → [B, D] ───────────────────
        pooled: torch.Tensor = self._masked_mean_pool(embeddings, padding_mask)

        # ── 2. Layer normalisation ─────────────────────────────────────
        pooled = self.layer_norm(pooled)                              # [B, D]

        # ── 3. MLP ────────────────────────────────────────────────────
        logit: torch.Tensor = self.mlp(pooled)                        # [B, 1]

        # ── 4. Softplus + de-normalise → raw frame prediction ─────────
        predicted_length: torch.Tensor = (
            self.output_activation(logit).squeeze(-1) * self.length_scale
        )  # [B]

        return predicted_length

    # ── Utilities ─────────────────────────────────────────────────────────────

    def predict_lengths(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper: run encode → round → clamp to discrete lengths.

        Returns
        -------
        ``LongTensor[B]`` — discrete predicted frame counts, always ≥ 1.
        """
        with torch.no_grad():
            lengths = self.forward(embeddings, padding_mask)
        return lengths.round().long().clamp(min=1)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total (or trainable-only) parameter count."""
        params = (
            self.parameters()
            if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        return (
            f"LengthPredictor("
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"dropout={self.dropout_p}, "
            f"length_scale={self.length_scale}, "
            f"params={self.num_parameters():,}"
            f")"
        )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    predictor = LengthPredictor(
        input_dim=768,
        hidden_dims=(256, 64),
        dropout=0.1,
        length_scale=100.0,
    ).to(device)

    print(predictor)
    print(f"  Total params     : {predictor.num_parameters():,}")
    print(f"  Trainable params : {predictor.num_parameters(trainable_only=True):,}")
    print()

    # ── Synthetic batch matching real SignVerse statistics ─────────────────
    B, T, D = 8, 24, 768
    torch.manual_seed(42)
    embeddings   = torch.randn(B, T, D, device=device)
    padding_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    # Simulate variable-length sentences (realistic padding distribution)
    real_lens = [24, 18, 12, 20, 8, 15, 22, 24]
    for i, real_len in enumerate(real_lens):
        if real_len < T:
            padding_mask[i, real_len:] = True
        # Zero-fill padding positions as TextEncoder does
        embeddings[i, real_len:] = 0.0

    print("── forward() pass ────────────────────────────────────────────")
    predictor.eval()
    with torch.no_grad():
        predicted = predictor(embeddings, padding_mask)

    print(f"  Output shape     : {tuple(predicted.shape)}  (expected [B={B}])")
    print(f"  Output dtype     : {predicted.dtype}")
    print(f"  Output device    : {predicted.device}")
    print(f"  All positive     : {(predicted > 0).all().item()}")
    print(f"  Predictions      : {predicted.round().long().tolist()}")
    print()

    # ── predict_lengths convenience method ────────────────────────────────
    print("── predict_lengths() ─────────────────────────────────────────")
    discrete = predictor.predict_lengths(embeddings, padding_mask)
    print(f"  Discrete lengths : {discrete.tolist()}")
    print(f"  All >= 1         : {(discrete >= 1).all().item()}")
    print()

    # ── Gradient flow (training mode) ─────────────────────────────────────
    print("── gradient flow ─────────────────────────────────────────────")
    predictor.train()
    target_frames = torch.tensor(
        [98.0, 120.0, 75.0, 110.0, 60.0, 88.0, 140.0, 95.0], device=device
    )
    pred_train = predictor(embeddings, padding_mask)
    pred_norm   = pred_train   / predictor.length_scale
    target_norm = target_frames / predictor.length_scale
    loss = F.huber_loss(pred_norm, target_norm)
    loss.backward()

    grad_norms = {
        name: p.grad.norm().item()
        for name, p in predictor.named_parameters()
        if p.grad is not None
    }
    print(f"  Loss             : {loss.item():.6f}")
    print(f"  Params with grad : {len(grad_norms)}")
    for name, gnorm in grad_norms.items():
        print(f"    {name:<40} grad_norm={gnorm:.6f}")
    assert all(g > 0 for g in grad_norms.values()), "Some gradients are zero!"
    print("  All gradients non-zero: PASSED")
    print()

    # ── Shape validation errors ────────────────────────────────────────────
    print("── input validation ──────────────────────────────────────────")
    predictor.eval()

    try:
        predictor(torch.randn(B, D, device=device), padding_mask)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Wrong emb ndim   : PASSED ({e})")

    try:
        predictor(embeddings, padding_mask[:, :5])
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Mask shape mismatch: PASSED ({e})")

    try:
        predictor(torch.randn(B, T, 512, device=device), padding_mask)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Wrong input_dim  : PASSED ({e})")

    print()

    # ── Edge cases ─────────────────────────────────────────────────────────
    print("── edge cases ────────────────────────────────────────────────")

    # Single-sample batch
    with torch.no_grad():
        single_pred = predictor(
            embeddings[:1], padding_mask[:1]
        )
    print(f"  Single-sample batch    : PASSED  shape={tuple(single_pred.shape)}")

    # All-padding row (degenerate — must not NaN/Inf)
    emb_degenerate  = torch.zeros(2, T, D, device=device)
    mask_degenerate = torch.ones(2, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        deg_pred = predictor(emb_degenerate, mask_degenerate)
    assert not torch.isnan(deg_pred).any(), "NaN in degenerate case!"
    assert not torch.isinf(deg_pred).any(), "Inf in degenerate case!"
    assert (deg_pred > 0).all(), "Prediction must remain positive!"
    print(f"  All-padding degenerate : PASSED  predictions={deg_pred.tolist()}")

    # Custom hidden dims
    custom = LengthPredictor(
        input_dim=768,
        hidden_dims=(512, 128, 32),
        dropout=0.0,
        length_scale=150.0,
    ).to(device)
    with torch.no_grad():
        custom_pred = custom(embeddings, padding_mask)
    assert custom_pred.shape == (B,)
    print(f"  Custom hidden_dims     : PASSED  shape={tuple(custom_pred.shape)}")
    print()

    print("All smoke tests passed.")