# =============================================================================
# trainer.py  —  Trainer class for DR binary classification
# =============================================================================
"""
Features
────────
• BCEWithLogitsLoss (BaseModel) / BCELoss (EnsembleModel) — auto-detected
• AdamW optimiser with decoupled weight decay
• ReduceLROnPlateau — LR halved when val AUC plateaus
• Early stopping on val AUC — saves only the best checkpoint
• Per-epoch history dict — loss + AUC for train and val splits
• evaluate()        — labelled loader → (AUC, labels, probs)
• test_inference()  — unlabelled loader → CSV of DR scores
• load_best()       — restore best checkpoint weights

Loss selection logic
─────────────────────
EnsembleModel.forward() applies sigmoid internally and returns averaged
probabilities in [0, 1].  BCEWithLogitsLoss expects *raw logits*, so the
Trainer detects an EnsembleModel instance and uses plain BCELoss instead.
For BaseModel and CustomCNN the raw logits go straight into BCEWithLogitsLoss.
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
from tqdm.notebook import tqdm
from colorama import Fore, Style, init

from utils.config import Config as cfg
from src.model import EnsembleModel
from src.model_components import FocalLoss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training / evaluation / inference workflow.

    Args:
        model:           nn.Module with output shape (B, 1).
                         EnsembleModel → BCELoss (probs).
                         Any other module → BCEWithLogitsLoss (logits).
        train_loader:    DataLoader for the training split.
        val_loader:      DataLoader for the validation split.
        pos_weight:      Scalar weight for the positive (DR) class.
                         ``None`` → computed automatically.
        lr:              Initial AdamW learning rate.
        weight_decay:    AdamW weight decay.
        device:          torch device string or object.
        checkpoint_dir:  Directory for saving the best checkpoint.
    """

    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        pos_weight:      Optional[float] = None,
        lr:              float           = cfg.learning_rate,
        weight_decay:    float           = cfg.weight_decay,
        device:          str             = cfg.device,
        checkpoint_dir:  str             = str(cfg.checkpoint_dir),
    ):
        self.device         = device
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.checkpoint_dir = str(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        init(autoreset=True)

        # ── Detect model output type ──────────────────────────────────────────
        #   EnsembleModel → probabilities → FocalLoss
        #   Everything else → raw logits  → BCEWithLogitsLoss
        self._is_ensemble = isinstance(model, EnsembleModel)

        # ── Loss function ─────────────────────────────────────────────────────
        if pos_weight is None:
            pos_weight = self._estimate_pos_weight(train_loader)
        pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        print(f"[Trainer] pos_weight = {pos_weight:.4f}")

        if self._is_ensemble:
            self.criterion = FocalLoss(weight=pw)
            print("[Trainer] Loss: FocalLoss (EnsembleModel returns probabilities)")
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print("[Trainer] Loss: BCEWithLogitsLoss  (model returns raw logits)")

        # ── Optimiser  ────────────────────────────────────────────────────────
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer   = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # ── LR scheduler ─────────────────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode    ="max",          # maximise val AUC
            factor  =cfg.lr_factor,
            patience=cfg.lr_patience,
        )

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
        }

        # ── Best-model tracking ───────────────────────────────────────────────
        self.best_val_auc  = 0.0
        self.best_epoch    = -1
        self._best_weights = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_pos_weight(loader: DataLoader) -> float:
        """Compute pos_weight = count_negative / count_positive from labels."""
        all_labels: list[int] = []
        for batch in loader:
            all_labels.extend(batch["label"].tolist())
        arr   = np.array(all_labels, dtype=int)
        n_neg = int((arr == 0).sum())
        n_pos = int((arr == 1).sum())
        if n_pos == 0:
            return 1.0
        pw = n_neg / n_pos
        print(f"[Trainer] Auto pos_weight: {n_neg} neg / {n_pos} pos = {pw:.3f}")
        return float(pw)

    def _forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified forward pass.

        Returns:
            (loss_input, probs) — both shape (B,)
              loss_input: passed directly to the loss function.
              probs:       sigmoid probabilities used for AUC.
        """
        raw = self.model(inputs).squeeze(1)     # (B,)
        if self._is_ensemble:
            # EnsembleModel already returns probabilities
            return raw, raw
        else:
            # BaseModel / CustomCNN return raw logits
            return raw, torch.sigmoid(raw)

    # ─────────────────────────────────────────────────────────────────────────
    # One epoch
    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> tuple[float, float]:
        """One full pass over a DataLoader.

        Returns:
            (mean_loss, roc_auc)
        """
        self.model.train(is_train)
        total_loss: float      = 0.0
        all_probs:  list[float] = []
        all_labels: list[int]  = []

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            tag       = "train" if is_train else "val  "
            batch_bar = tqdm(loader, desc=f"[{tag}]", leave=False)
            for batch in batch_bar:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                loss_input, probs = self._forward(inputs)
                loss = self.criterion(loss_input, labels)

                if is_train:
                    loss.backward()
                    # Gradient clipping for stability (important for ViT heads)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
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
        num_epochs:      int = cfg.num_epochs,
        early_stopping:  int = cfg.early_stopping,
        checkpoint_name: str = "best_model.pth",
    ) -> dict:
        """Train the model, applying early stopping on validation AUC.

        Args:
            num_epochs:      Maximum number of epochs to run.
            early_stopping:  Stop if val AUC does not improve for this many
                             consecutive epochs.
            checkpoint_name: Filename for the saved best checkpoint.

        Returns:
            History dict with keys: train_loss, val_loss, train_auc, val_auc.
        """
        since             = time.time()
        epochs_no_improve = 0
        checkpoint_path   = os.path.join(self.checkpoint_dir, checkpoint_name)

        print(f"\n{'='*60}")
        print(f"  Training on {self.device}  |  max epochs: {num_epochs}")
        print(f"  Early-stop patience: {early_stopping}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_loss, train_auc = self._run_epoch(self.train_loader, is_train=True)
            val_loss,   val_auc   = self._run_epoch(self.val_loader,   is_train=False)

            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                f"Train loss: {train_loss:.4f}  AUC: {train_auc:.4f}  |  "
                f"Val   loss: {val_loss:.4f}  AUC: {val_auc:.4f}  |  "
                f"LR: {current_lr:.2e}  [{elapsed:.1f}s]"
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc  = val_auc
                self.best_epoch    = epoch + 1
                self._best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self._best_weights, checkpoint_path)
                print(
                    f"  ✓ New best val AUC: {val_auc:.4f}  "
                    f"(checkpoint → {checkpoint_path})"
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(
                        f"\n[EarlyStopping] No improvement for {early_stopping} "
                        f"epochs. Stopping at epoch {epoch+1}."
                    )
                    break

        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

        total_time = time.time() - since
        print(
            f"\n{'='*60}\n"
            f"  Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s\n"
            f"  Best epoch: {self.best_epoch}  |  Best val AUC: {self.best_val_auc:.4f}\n"
            f"{'='*60}"
        )
        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation & inference
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        loader: DataLoader,
    ) -> tuple[float, np.ndarray, np.ndarray]:
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
        auc        = metrics.roc_auc_score(labels_arr, probs_arr)
        return auc, labels_arr, probs_arr

    def test_inference(
        self,
        test_loader: DataLoader,
        output_path: str,
        num_tta: int,
    ) -> np.ndarray:
        """Run inference on the unlabelled test set and save scores to CSV.

        Produces a 1000×1 CSV (one DR probability per row) for Codabench.

        Args:
            test_loader:  DataLoader for the test split (labels are -1).
            output_path:  Destination path for the output CSV.
            num_tta:      Number of TTA iterations if 0 doesn't apply.

        Returns:
            numpy array of shape (N, 1) with DR scores in [0, 1].
        """
        self.model.eval()

        all_tta_scores = np.zeros((len(test_loader), 1))
        all_scores: list[float] = []
        num_tta = max(1, num_tta)

        print(f"[test_inference] Start inferene with {num_tta} iterations...")

        for i in range(num_tta):
            current_pass_scores = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["image"].to(self.device)
                    if self._is_ensemble:
                        probs = self.model(inputs).squeeze(1)
                    else:
                        probs = torch.sigmoid(self.model(inputs)).squeeze(1)
                    current_pass_scores.extend(probs.cpu().tolist())
            all_tta_scores += np.array(current_pass_scores).reshape(-1, 1)

        final_scores = all_tta_scores / num_tta
        with open(output_path, mode="w", newline="") as f:
            csv.writer(f).writerows(final_scores)

        print(
            f"[Trainer] Saved {len(final_scores)} scores → {output_path}\n"
            f"          Range: [{final_scores.min():.4f}, {final_scores.max():.4f}]"
        )
        return final_scores

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint utilities
    # ─────────────────────────────────────────────────────────────────────────

    def load_best(self, checkpoint_name: str = "best_model.pth"):
        """Load the best checkpoint back into the model."""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            print(f"[Trainer] Loaded checkpoint: {path}")
        else:
            print(f"[Trainer] WARNING: checkpoint not found at {path}")

    def save_checkpoint(self, name: str = "manual_checkpoint.pth"):
        """Manually save the current model state."""
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"[Trainer] Checkpoint saved → {path}")