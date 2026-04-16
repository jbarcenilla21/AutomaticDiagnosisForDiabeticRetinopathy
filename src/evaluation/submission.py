"""Submission generation for Codabench.

Codabench expects a ZIP containing two files:
    output_custom.csv  — 1000x1 DR scores from the Custom model
    output_ft.csv      — 1000x1 DR scores from the Fine-tune model

Workflow after training:
    1. test_model()           -> generate 1000 scores for the test set
    2. save_strategy_results()-> persist scores to results/{strategy}_results/{run_name}.csv
    3. generate_submission()  -> combine both strategies (most recent if available,
                                 random otherwise) -> ZIP
"""

import csv
import os
from zipfile import ZipFile

import numpy as np
import torch


_RESULTS_FOLDER = {
    'custom':    'custom_results',
    'fine_tune': 'fine_tune_results',
}


# ─── Inference ───────────────────────────────────────────────────────────────

def test_model(model, test_loader, device):
    """Run inference on the test set and return a (N, 1) numpy array of scores.

    Args:
        model:       Trained nn.Module (already on device).
        test_loader: DataLoader for the test set (no labels required).
        device:      torch.device.

    Returns:
        np.ndarray of shape (N, 1) with DR scores in [0, 1].
    """
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for sample in test_loader:
            inputs = sample['image'].to(device).float()
            outputs = model(inputs)
            outputs = outputs.view(-1, 1)
            all_outputs.append(outputs.cpu().numpy())

    return np.concatenate(all_outputs, axis=0)


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_strategy_results(outputs, strategy, results_dir, run_name):
    """Save the (N, 1) score array as {run_name}.csv in the strategy results folder.

    Each call produces a new named file — previous runs are NOT overwritten.

    Args:
        outputs:     np.ndarray (N, 1).
        strategy:    'custom' or 'fine_tune'.
        results_dir: root results directory on Drive.
        run_name:    descriptive name for this run (no extension), e.g.
                     'custom_auc0.742_20260416_143022'.

    Returns:
        Path to the saved CSV.
    """
    folder = os.path.join(results_dir, _RESULTS_FOLDER[strategy])
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f'{run_name}.csv')

    _write_csv(csv_path, outputs)
    print(f'Saved {strategy} scores -> {csv_path}')
    return csv_path


def _most_recent_csv(strategy, results_dir):
    """Return the path to the most recently modified CSV in the strategy folder, or None."""
    folder = os.path.join(results_dir, _RESULTS_FOLDER[strategy])
    if not os.path.isdir(folder):
        return None
    csvs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.csv')
    ]
    if not csvs:
        return None
    return max(csvs, key=os.path.getmtime)


def _load_csv(csv_path):
    scores = []
    with open(csv_path, 'r') as f:
        for row in csv.reader(f):
            scores.append([float(row[0])])
    return np.array(scores)


def _load_or_random(strategy, results_dir, n=1000, current_csv_path=None):
    """Return (N, 1) scores: from current_csv_path, most recent file, or random."""
    if current_csv_path and os.path.exists(current_csv_path):
        print(f'Using current run scores for {strategy}: {current_csv_path}')
        return _load_csv(current_csv_path)

    recent = _most_recent_csv(strategy, results_dir)
    if recent:
        print(f'Using most recent {strategy} scores: {recent}')
        return _load_csv(recent)

    print(f'No {strategy} results found — using random scores.')
    return np.random.rand(n, 1)


# ─── Submission ZIP ──────────────────────────────────────────────────────────

def generate_submission(current_strategy, results_dir, current_csv_path, n_test=1000):
    """Build codabench_submission.zip combining both strategies.

    Uses current_csv_path for the strategy just trained.
    For the other strategy, loads the most recently saved CSV (or random if none).
    The ZIP is saved to results_dir/Submissions/.

    Args:
        current_strategy: the strategy just trained ('custom' or 'fine_tune').
        results_dir:      root results directory on Drive.
        current_csv_path: path to the CSV saved by save_strategy_results() this run.
        n_test:           number of test samples (default 1000).

    Returns:
        Path to the generated ZIP file.
    """
    other = 'fine_tune' if current_strategy == 'custom' else 'custom'

    current_scores = _load_or_random(current_strategy, results_dir, n_test, current_csv_path)
    other_scores   = _load_or_random(other,             results_dir, n_test)

    submissions_dir = os.path.join(results_dir, 'Submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    custom_path = os.path.join(submissions_dir, 'output_custom.csv')
    ft_path     = os.path.join(submissions_dir, 'output_ft.csv')

    if current_strategy == 'custom':
        _write_csv(custom_path, current_scores)
        _write_csv(ft_path,     other_scores)
    else:
        _write_csv(custom_path, other_scores)
        _write_csv(ft_path,     current_scores)

    zip_path = os.path.join(submissions_dir, 'codabench_submission.zip')
    with ZipFile(zip_path, 'w') as zf:
        zf.write(custom_path, arcname='output_custom.csv')
        zf.write(ft_path,     arcname='output_ft.csv')

    print(f'Submission ZIP -> {zip_path}')
    return zip_path


def _write_csv(path, scores):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(scores.tolist())
