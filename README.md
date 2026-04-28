# Automatic Diagnosis for Diabetic Retinopathy

Binary classification of diabetic retinopathy (DR) from color fundus photographs using CNNs in PyTorch.  
UC3M Master in Machine Learning for Health — Computer Vision Lab Session 5.

## Goal

Develop a model that takes a retinal fundus image and outputs a binary diagnosis:

| Label | Meaning |
|-------|---------|
| 0 | No DR |
| 1 | DR (any stage) |

Two submission tracks on Codabench:
- **Custom**: CNN built entirely from scratch — no pretrained weights allowed.
- **Fine-tuning**: Pretrained torchvision backbone fine-tuned for DR.

Evaluation metric: **AUC** (area under the ROC curve).

---

## Repository Structure

```
├── LS5_CV_CNNs_Retinopathy.ipynb   # Original lab notebook (deliverable)
├── train_colab.ipynb               # Clean training notebook for Google Colab Pro
├── src/                            # All Python modules — upload to Drive to use in Colab
│   ├── data/
│   │   ├── dataset.py              # RetinopathyDataset (torch.utils.data.Dataset)
│   │   └── transforms.py           # Custom transforms + torchvision wrappers + pipelines
│   ├── model/
│   │   ├── custom_net.py           # CustomNet — 4-block CNN built from scratch
│   │   └── fine_tune_net.py        # build_fine_tune_model() for alexnet/resnet/vgg/efficientnet
│   ├── training/
│   │   ├── config.py               # Hyperparameters
│   │   └── trainer.py              # train_model() with best-checkpoint saving
│   └── evaluation/
│       ├── metrics.py              # compute_auc()
│       └── submission.py           # test_model(), save_strategy_results(), generate_submission()
├── md/                             # Documentation
├── data/                           # Dataset — not tracked by git
│   ├── images/                     # 3500 retinal fundus images (.jpg)
│   ├── train.csv                   # 2000 samples
│   ├── val.csv                     # 500 samples
│   └── test.csv                    # 1000 samples (labels = -1)
├── outputs/                        # Submission files — not tracked by git
└── requirements.txt
```

---

## Colab Setup

The training notebook (`train_colab.ipynb`) runs entirely on Google Colab Pro with data stored in Drive.

**Expected Drive layout:**
```
MyDrive/Computer Vision/AutomaticDiagnosisForDiabeticRetinopathy/
  data/               ← images + CSVs
  src/                ← upload this folder from the repo
  models/
    custom/           ← one .pth per completed run
    fine_tune/        ← one .pth per completed run
  results/
    custom_results/   ← one .csv per completed run
    fine_tune_results/
    Submissions/      ← codabench_submission.zip (always up to date)
```

**Steps:**
1. Upload `src/` to Drive at the path above.
2. Open `train_colab.ipynb` in Colab.
3. Set `STRATEGY = 'custom'` or `'fine_tune'` (and `BACKBONE` if fine-tuning).
4. Run all cells.

**Troubleshooting — `ModuleNotFoundError: No module named 'data'`:**  
The paths cell must run before the imports cell. If the error persists, verify the Drive path:
```python
print(SRC_DIR)
print(os.path.exists(SRC_DIR))
print(os.listdir(SRC_DIR))
```
If `SRC_DIR` does not exist, the folder name in Drive does not exactly match `DRIVE_BASE` in the notebook — update it to match.

---

## Model & Results Management

- **During training**: best checkpoint is saved to `_temp_best.pth` (overwritten on each improvement).
- **After training**: temp file is renamed to a permanent descriptive name:  
  `{strategy}[_{backbone}]_auc{val_auc}_{YYYYMMDD_HHMMSS}.pth`  
  Example: `fine_tune_alexnet_auc0.8210_20260416_150311.pth`
- **Scores**: same naming convention, saved as `.csv` in the corresponding results folder.
- **Submission ZIP**: always regenerated after each run. Uses the current run's scores for the trained strategy and the most recently saved CSV for the other (random scores if none exist yet).

---

## Local Setup

```bash
pip install -r requirements.txt
```

Place the extracted dataset under `data/` — see [Files](md/Files.md) for the expected structure.

---

## Documentation

| File | Contents |
|------|----------|
| [Lab session description and database](md/Lab%20session%20description%20and%20database.md) | Problem context, dataset format and splits |
| [Design and implementation of the diagnosis system](md/Design%20and%20implementation%20of%20the%20diagnosis%20system.md) | Pipeline: data loading, custom CNN, fine-tuning |
| [Evaluation Metric AUC](md/Evaluation%20Metric%20AUC.md) | AUC metric, ROC curve, scikit-learn usage |
| [Files](md/Files.md) | Downloads, local folder structure, CSV format |
| [Terms](md/Terms.md) | Evaluation criteria, challenge rules, submission details |
