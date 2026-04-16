"""Training loop for DR classification models.

During training, the running best checkpoint is kept at `temp_save_path`
(overwritten on every improvement). This file is temporary — the caller is
responsible for renaming it to a permanent, meaningful path after training.

Returns (model, best_auc) so the caller can build a descriptive filename.
"""

import copy
import os
import time

import numpy as np
import torch
from sklearn import metrics


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25,
    temp_save_path=None,
):
    """Train and validate a model, keeping the best checkpoint at temp_save_path.

    Args:
        model:          nn.Module to train.
        criterion:      Loss function (e.g. nn.BCELoss).
        optimizer:      Optimizer instance.
        scheduler:      LR scheduler instance.
        dataloaders:    Dict with 'train' and 'val' DataLoader.
        dataset_sizes:  Dict with 'train' and 'val' dataset lengths.
        device:         torch.device.
        num_epochs:     Number of training epochs.
        temp_save_path: Path for the running-best checkpoint during training.
                        Overwritten on every improvement. If None, no saving.

    Returns:
        (model, best_auc): model loaded with best weights; best val AUC achieved.
    """
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_auc     = 0.0
    best_epoch   = -1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            n = dataset_sizes[phase]
            all_outputs = np.zeros(n, dtype=float)
            all_labels  = np.zeros(n, dtype=int)
            running_loss = 0.0
            cursor = 0

            for sample in dataloaders[phase]:
                inputs = sample['image'].to(device).float()
                labels = sample['label'].to(device).float()
                bs = labels.shape[0]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.flatten()
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * bs
                all_outputs[cursor:cursor + bs] = outputs.detach().cpu().numpy()
                all_labels[cursor:cursor + bs]  = labels.cpu().numpy().astype(int)
                cursor += bs

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc  = metrics.roc_auc_score(all_labels, all_outputs)
            print(f'  {phase:5s}  loss: {epoch_loss:.4f}  AUC: {epoch_auc:.4f}')

            if phase == 'val' and epoch_auc > best_auc:
                best_auc     = epoch_auc
                best_epoch   = epoch
                best_weights = copy.deepcopy(model.state_dict())
                if temp_save_path:
                    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
                    torch.save(model.state_dict(), temp_save_path)
                    print(f'  -> New best (AUC={best_auc:.4f}) saved to temp checkpoint')

        print()

    elapsed = time.time() - since
    print(f'Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    print(f'Best val AUC: {best_auc:.4f} at epoch {best_epoch}')

    model.load_state_dict(best_weights)
    return model, best_auc
