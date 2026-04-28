# =============================================================================
# trainer.py  —  Trainer class for DR binary classification
# =============================================================================
"""
Features
────────
• FocalLoss for both BaseModel (from_logits=True) and EnsembleModel
  (from_logits=False) — selected automatically based on model type.
• AdamW optimiser with decoupled weight decay.
• ReduceLROnPlateau — LR scaled when val AUC plateaus.
• Early stopping on val AUC — saves only the best checkpoint.
• Per-epoch history — loss, AUC, and accuracy for both train and val splits.
• evaluate()        — labelled loader → (AUC, accuracy, labels, probs).
• test_inference()  — unlabelled loader → CSV of DR scores (with optional TTA).
• load_best()       — restore best checkpoint weights.

Loss selection logic
─────────────────────
EnsembleModel.forward() applies sigmoid internally and returns probabilities
in [0, 1].  FocalLoss with from_logits=False handles this case.
BaseModel / Custom_VGG return raw logits → FocalLoss with from_logits=True.
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

from utils.config import Config as cfg
from src.model import EnsembleModel
from src.model_components import FocalLoss


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Encapsulates the full training / evaluation / inference workflow.

    Args:
        model:           nn.Module outputting shape (B, 1).
                         EnsembleModel → probabilities → FocalLoss(from_logits=False).
                         BaseModel / Custom_VGG → raw logits → FocalLoss(from_logits=True).
        train_loader:    DataLoader for the training split.
        val_loader:      DataLoader for the validation split.
        lr:              Initial AdamW learning rate.
        weight_decay:    AdamW weight decay.
        device:          torch device string.
        checkpoint_dir:  Directory for the best checkpoint.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        lr:             float = cfg.learning_rate,
        weight_decay:   float = cfg.weight_decay,
        device:         str   = cfg.device,
        checkpoint_dir: str   = str(cfg.checkpoint_dir),
    ):
        self.device         = device
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.checkpoint_dir = str(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Detect model output type ──────────────────────────────────────────
        # EnsembleModel → forward() returns probabilities (sigmoid inside).
        # Everything else → forward() returns raw logits.
        self._is_ensemble = isinstance(model, EnsembleModel)

        # ── Loss function ─────────────────────────────────────────────────────
        # FocalLoss handles class imbalance by down-weighting easy examples.
        # from_logits controls whether the input to the loss is a logit or prob.
        if self._is_ensemble:
            self.criterion = FocalLoss(
                alpha       = cfg.focal_alpha,
                gamma       = cfg.focal_gamma,
                from_logits = False,   # EnsembleModel outputs probabilities
            )
            print(
                f"[Trainer] Loss: FocalLoss(from_logits=False)  "
                f"[α={cfg.focal_alpha}, γ={cfg.focal_gamma}]  "
                f"← EnsembleModel returns probabilities"
            )
        else:
            self.criterion = FocalLoss(
                alpha       = cfg.focal_alpha,
                gamma       = cfg.focal_gamma,
                from_logits = True,    # BaseModel / Custom_VGG return raw logits
            )
            print(
                f"[Trainer] Loss: FocalLoss(from_logits=True)  "
                f"[α={cfg.focal_alpha}, γ={cfg.focal_gamma}]  "
                f"← model returns raw logits"
            )

        # ── Optimiser ─────────────────────────────────────────────────────────
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer   = torch.optim.AdamW(
            trainable_params,
            lr           = lr,
            weight_decay = weight_decay,
        )

        # ── LR scheduler ──────────────────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "max",          # maximise val AUC
            factor   = cfg.lr_factor,
            patience = cfg.lr_patience,
        )

        # ── History ───────────────────────────────────────────────────────────
        # Tracks loss, AUC, and accuracy for both splits at every epoch.
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss":  [],
            "train_auc":  [], "val_auc":   [],
            "train_acc":  [], "val_acc":   [],
        }

        # ── Best-model tracking ───────────────────────────────────────────────
        self.best_val_auc  = 0.0
        self.best_epoch    = -1
        self._best_weights = None

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified forward pass returning (loss_input, probs).

        Returns:
            loss_input: Tensor passed directly to the loss function.
                        Logits for BaseModel; probabilities for EnsembleModel.
            probs:      Sigmoid probabilities used for AUC and accuracy.
        """
        raw = self.model(inputs)
        if self._is_ensemble:
            # EnsembleModel already applies sigmoid and returns (B, 1)
            # Keep the (B, 1) shape for the loss, but flatten probs for metrics.
            return raw, raw.squeeze(1)
        else:
            raw = raw.squeeze(1)
            return raw, torch.sigmoid(raw)

    # =========================================================================
    # Single epoch
    # =========================================================================

    def _run_epoch(
        self, loader: DataLoader, is_train: bool
    ) -> tuple[float, float, float]:
        """One full pass over a DataLoader.

        Returns:
            (mean_loss, roc_auc, accuracy)
        """
        self.model.train(is_train)
        total_loss:  float       = 0.0
        all_probs:   list[float] = []
        all_labels:  list[int]   = []
        all_preds:   list[int]   = []

        context = torch.enable_grad() if is_train else torch.no_grad()
        tag     = "train" if is_train else "val  "

        with context:
            bar = tqdm(loader, desc=f"[{tag}]", leave=False)
            for batch in bar:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                loss_input, probs = self._forward(inputs)
                loss = self.criterion(loss_input, labels)

                if is_train:
                    loss.backward()
                    # Gradient clipping: prevents exploding gradients in ViT heads
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)

                probs_np = probs.detach().cpu().numpy()
                all_probs.extend(probs_np.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                # Binary prediction at threshold 0.5
                all_preds.extend((probs_np >= 0.5).astype(int).tolist())

        n          = max(len(all_labels), 1)
        mean_loss  = total_loss / n

        # AUC — falls back to 0.5 if only one class is present (edge case)
        try:
            auc = metrics.roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        # Accuracy — fraction of correctly classified samples
        acc = metrics.accuracy_score(all_labels, all_preds)

        return mean_loss, auc, acc

    # =========================================================================
    # Main training loop
    # =========================================================================

    def fit(
        self,
        num_epochs:      int = cfg.num_epochs,
        early_stopping:  int = cfg.early_stopping,
        checkpoint_name: str = "best_model.pth",
    ) -> dict:
        """Train the model, applying early stopping on validation AUC.

        Args:
            num_epochs:      Maximum number of epochs.
            early_stopping:  Stop if val AUC does not improve for this many epochs.
            checkpoint_name: Filename for the saved best checkpoint.

        Returns:
            History dict: train_loss, val_loss, train_auc, val_auc, train_acc, val_acc.
        """
        since             = time.time()
        epochs_no_improve = 0
        checkpoint_path   = os.path.join(self.checkpoint_dir, checkpoint_name)

        print(f"\n{'='*65}")
        print(f"  Training on {self.device}  |  max epochs: {num_epochs}")
        print(f"  Early-stop patience: {early_stopping}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'='*65}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_loss, train_auc, train_acc = self._run_epoch(self.train_loader, is_train=True)
            val_loss,   val_auc,   val_acc   = self._run_epoch(self.val_loader,   is_train=False)

            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                f"Train  loss={train_loss:.4f}  AUC={train_auc:.4f}  acc={train_acc:.3f}  |  "
                f"Val    loss={val_loss:.4f}  AUC={val_auc:.4f}  acc={val_acc:.3f}  |  "
                f"LR={current_lr:.2e}  [{elapsed:.1f}s]"
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc  = val_auc
                self.best_epoch    = epoch + 1
                self._best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self._best_weights, checkpoint_path)
                print(f"  ✓ New best val AUC: {val_auc:.4f}  (checkpoint → {checkpoint_path})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    print(
                        f"\n[EarlyStopping] No improvement for {early_stopping} "
                        f"epochs — stopping at epoch {epoch + 1}."
                    )
                    break

        # Restore best weights before returning
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

        total_time = time.time() - since
        print(
            f"\n{'='*65}\n"
            f"  Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s\n"
            f"  Best epoch: {self.best_epoch}  |  Best val AUC: {self.best_val_auc:.4f}\n"
            f"{'='*65}"
        )
        return self.history

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(
        self, loader: DataLoader
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Run inference on a labelled DataLoader.

        Returns:
            (auc, accuracy, labels_array, probs_array)
        """
        self.model.eval()
        all_probs:  list[float] = []
        all_labels: list[int]   = []
        all_preds:  list[int]   = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                _, probs = self._forward(inputs)
                probs_np = probs.cpu().numpy()
                all_probs.extend(probs_np.tolist())
                all_labels.extend(labels.tolist())
                all_preds.extend((probs_np >= 0.5).astype(int).tolist())

        labels_arr = np.array(all_labels, dtype=int)
        probs_arr  = np.array(all_probs,  dtype=float)
        auc        = metrics.roc_auc_score(labels_arr, probs_arr)
        acc        = metrics.accuracy_score(labels_arr, all_preds)
        return auc, acc, labels_arr, probs_arr

    # =========================================================================
    # Test inference (with TTA)
    # =========================================================================

    def test_inference(
        self,
        test_loader: DataLoader,
        output_path: str,
        num_tta:     int = cfg.num_tta,
    ) -> np.ndarray:
        """Run inference on the unlabelled test set and save scores to CSV.

        Produces a 1000×1 CSV (one DR probability per row) for Codabench.

        TTA strategy: the DataLoader is iterated ``num_tta`` times.  Since the
        DataLoader uses stochastic transforms (build_tta_transforms), each pass
        produces different augmentations — their mean is the final prediction.
        If ``num_tta == 1`` no augmentation is applied (deterministic eval).

        Args:
            test_loader: DataLoader for the test split (labels are -1).
            output_path: Destination path for the output CSV.
            num_tta:     Number of TTA iterations.

        Returns:
            numpy array of shape (N, 1) with DR scores in [0, 1].
        """
        self.model.eval()
        num_tta = max(1, num_tta)

        print(f"[test_inference] Running {num_tta} TTA pass(es) over {len(test_loader.dataset)} images ...")

        # Accumulate scores over all TTA passes
        all_tta_scores: Optional[np.ndarray] = None

        for pass_idx in range(num_tta):
            pass_scores: list[float] = []

            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["image"].to(self.device)

                    if self._is_ensemble:
                        probs = self.model(inputs).squeeze(1)
                    else:
                        probs = torch.sigmoid(self.model(inputs)).squeeze(1)

                    pass_scores.extend(probs.cpu().tolist())

            scores_arr = np.array(pass_scores, dtype=float).reshape(-1, 1)

            # Initialise on the first pass, then accumulate
            if all_tta_scores is None:
                all_tta_scores = scores_arr
            else:
                all_tta_scores = all_tta_scores + scores_arr

            if (pass_idx + 1) % max(num_tta // 5, 1) == 0 or pass_idx == num_tta - 1:
                print(f"  Pass {pass_idx + 1}/{num_tta} done.")

        # Average across passes
        final_scores = all_tta_scores / num_tta

        # Save to CSV (one score per row, no header — Codabench format)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, mode="w", newline="") as f:
            csv.writer(f).writerows(final_scores)

        print(
            f"[Trainer] Saved {len(final_scores)} scores → {output_path}\n"
            f"          Range: [{final_scores.min():.4f}, {final_scores.max():.4f}]  "
            f"Mean: {final_scores.mean():.4f}"
        )
        return final_scores

    # =========================================================================
    # Checkpoint utilities
    # =========================================================================

    def load_best(self, checkpoint_name: str = "best_model.pth"):
        """Load the best checkpoint back into the model."""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[Trainer] Loaded checkpoint: {path}")
        else:
            print(f"[Trainer] WARNING: checkpoint not found at {path}")

    def save_checkpoint(self, name: str = "manual_checkpoint.pth"):
        """Manually save the current model state."""
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"[Trainer] Checkpoint saved → {path}")