# =============================================================================
# trainer.py  —  Trainer class for DR binary classification
# =============================================================================
"""
Features
────────
• BCEWithLogitsLoss with pos_weight  – down-weights majority class errors
• AdamW optimiser                    – decoupled weight decay
• ReduceLROnPlateau                  – LR halved when val AUC plateaus
• Early stopping on val AUC          – saves only the best checkpoint
• Per-epoch history dict             – loss + AUC for both splits
• test_inference()                   – writes 1000×1 CSV for Codabench

Design note on the EnsembleModel
──────────────────────────────────
EnsembleModel.forward() applies sigmoid internally and returns averaged
probabilities in [0,1].  BCEWithLogitsLoss expects raw logits, so the Trainer
detects an EnsembleModel and uses plain BCELoss instead (probabilities → BCE).
For BaseModel the raw logits go straight into BCEWithLogitsLoss as intended.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader

from utils.config import Config as cfg
from src.model import EnsembleModel
from tqdm.notebook import tqdm 
import time
from colorama import Fore, Style, init



# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training / evaluation / inference workflow.

    Args:
        model:         nn.Module with output shape (B, 1).
        train_loader:  DataLoader for the training split.
        val_loader:    DataLoader for the validation split.
        pos_weight:    Scalar weight for the positive (DR) class in the loss.
                       Use ``None`` to compute it automatically from the training
                       set label distribution.
        lr:            Initial AdamW learning rate.
        weight_decay:  AdamW weight decay.
        device:        torch device.
        checkpoint_dir: Where to save the best checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pos_weight: Optional[float] = None,
        lr: float = cfg.learning_rate,
        weight_decay: float = cfg.weight_decay,
        device: torch.device = cfg.device,
        checkpoint_dir: str = cfg.checkpoint_dir,
    ):
        self.model           = model.to(device)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.device          = device
        self.checkpoint_dir  = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Determine whether the model outputs logits or probabilities ───────
        self._is_ensemble = isinstance(model, EnsembleModel)

        init(autoreset=True)

        # ── Loss ──────────────────────────────────────────────────────────────
        if pos_weight is None:
            pos_weight = self._estimate_pos_weight(train_loader)
        pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        print(f"[Trainer] pos_weight = {pos_weight:.4f}")

        if self._is_ensemble:
            # EnsembleModel outputs probabilities → use BCELoss
            self.criterion = nn.BCELoss(weight=pw)
            print("[Trainer] Loss: BCELoss (ensemble outputs probabilities)")
        else:
            # BaseModel outputs raw logits → BCEWithLogitsLoss (numerically stable)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print("[Trainer] Loss: BCEWithLogitsLoss (raw logits)")

        # ── Optimiser & scheduler ─────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",                      # monitor val AUC (higher = better)
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
        )

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
        }

        # ── Best-model state ──────────────────────────────────────────────────
        self.best_val_auc   = 0.0
        self.best_epoch     = -1
        self._best_weights  = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_pos_weight(loader: DataLoader) -> float:
        """Compute pos_weight = count_negative / count_positive from labels."""
        all_labels: list[int] = []
        for batch in loader:
            all_labels.extend(batch["label"].tolist())
        arr     = np.array(all_labels, dtype=int)
        n_neg   = (arr == 0).sum()
        n_pos   = (arr == 1).sum()
        if n_pos == 0:
            return 1.0
        pw = n_neg / n_pos
        print(f"[Trainer] Auto pos_weight: {n_neg} neg / {n_pos} pos = {pw:.3f}")
        return float(pw)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns probabilities in [0, 1] for AUC accumulation
        and the tensor needed by the loss (logit or probability, depending on
        the model type).

        Returns:
            (loss_input, probs)  both shape (B,)
        """
        raw = self.model(inputs).squeeze(1)   # (B,)
        if self._is_ensemble:
            return raw, raw                   # already probabilities
        else:
            return raw, torch.sigmoid(raw)    # logits for loss, probs for AUC

    # ─────────────────────────────────────────────────────────────────────────
    # One epoch
    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> tuple[float, float]:
        """Run one full pass over a DataLoader.

        Returns:
            (mean_loss, auc)
        """
        self.model.train(is_train)
        total_loss     = 0.0
        all_probs: list[float] = []
        all_labels: list[int]  = []

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            tag = "train" if is_train else "val"
            batch_bar = tqdm(loader, bar_format=cfg.bar_format, ascii="░█", 
                             desc=f"{Fore.WHITE}{tag}",)
            for batch in batch_bar:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

                if is_train:
                    self.optimizer.zero_grad()

                loss_input, probs = self._forward(inputs)
                loss = self.criterion(loss_input, labels)

                if is_train:
                    loss.backward()
                    # Gradient clipping for stability
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss  += loss.item() * inputs.size(0)
                all_probs.extend(probs.detach().cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        mean_loss = total_loss / max(len(all_labels), 1)

        try:
            auc = metrics.roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5   # only one class in this split (rare edge case)

        return mean_loss, auc

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        num_epochs: int = cfg.num_epochs,
        early_stopping: int = cfg.early_stopping,
        checkpoint_name: str = "best_model.pth",
    ) -> dict:
        """Train the model, applying early stopping on validation AUC.

        Args:
            num_epochs:         Maximum number of epochs.
            early_stopping:     Stop if val AUC does not improve for this
                                many consecutive epochs.
            checkpoint_name:    Filename for the saved best checkpoint.

        Returns:
            History dict with lists of train/val loss and AUC.
        """
        since               = time.time()
        epochs_no_improve   = 0
        checkpoint_path     = os.path.join(self.checkpoint_dir, checkpoint_name)

        print(f"\n{'='*60}")
        print(f"  Training on {self.device}  |  max epochs: {num_epochs}")
        print(f"  Early stop patience: {early_stopping}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # ── Train ────────────────────────────────────────────────────────
            train_loss, train_auc = self._run_epoch(self.train_loader, is_train=True)

            # ── Validate ─────────────────────────────────────────────────────
            val_loss, val_auc = self._run_epoch(self.val_loader, is_train=False)

            # ── LR scheduler ─────────────────────────────────────────────────
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Log ──────────────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                f"Train loss: {train_loss:.4f}  AUC: {train_auc:.4f}  |  "
                f"Val loss: {val_loss:.4f}  AUC: {val_auc:.4f}  |  "
                f"LR: {current_lr:.2e}  [{elapsed:.1f}s]"
            )

            # ── Best model ───────────────────────────────────────────────────
            if val_auc > self.best_val_auc:
                self.best_val_auc  = val_auc
                self.best_epoch    = epoch + 1
                self._best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self._best_weights, checkpoint_path)
                print(f"  ✓ New best val AUC: {val_auc:.4f}  (saved → {checkpoint_path})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(
                        f"\n[EarlyStopping] No val AUC improvement for "
                        f"{early_stopping} epochs. Stopping at epoch {epoch+1}."
                    )
                    break

        # ── Restore best weights ──────────────────────────────────────────────
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

        total_time = time.time() - since
        print(
            f"\n{'='*60}\n"
            f"  Training complete in {total_time//60:.0f}m {total_time%60:.0f}s\n"
            f"  Best epoch: {self.best_epoch}  |  Best val AUC: {self.best_val_auc:.4f}\n"
            f"{'='*60}"
        )
        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Run inference on a labelled DataLoader.

        Returns:
            (auc, labels_array, probs_array)
        """
        self.model.eval()
        all_probs:  list[float] = []
        all_labels: list[int]   = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                _, probs = self._forward(inputs)
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.tolist())

        labels_arr = np.array(all_labels, dtype=int)
        probs_arr  = np.array(all_probs,  dtype=float)
        auc = metrics.roc_auc_score(labels_arr, probs_arr)
        return auc, labels_arr, probs_arr

    def test_inference(
        self,
        test_loader: DataLoader,
        output_path: str,
    ) -> np.ndarray:
        """Run inference on the unlabelled test set and save scores to CSV.

        The output is a 1000×1 CSV (one DR score per row) as required by
        the Codabench submission format.

        Args:
            test_loader:  DataLoader for the test split (labels are -1).
            output_path:  Path for the output CSV file.

        Returns:
            numpy array of shape (N, 1) with DR scores in [0, 1].
        """
        self.model.eval()
        all_scores: list[float] = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"].to(self.device)
                if self._is_ensemble:
                    probs = self.model(inputs).squeeze(1)
                else:
                    probs = torch.sigmoid(self.model(inputs)).squeeze(1)
                all_scores.extend(probs.cpu().tolist())

        scores = np.array(all_scores, dtype=float).reshape(-1, 1)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(scores)

        print(
            f"[Trainer] Saved {len(scores)} scores → {output_path}\n"
            f"          Range: [{scores.min():.4f}, {scores.max():.4f}]"
        )
        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint utilities
    # ─────────────────────────────────────────────────────────────────────────

    def load_best(self, checkpoint_name: str = "best_model.pth"):
        """Load the best checkpoint back into the model."""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[Trainer] Loaded checkpoint: {path}")
        else:
            print(f"[Trainer] WARNING: checkpoint not found at {path}")