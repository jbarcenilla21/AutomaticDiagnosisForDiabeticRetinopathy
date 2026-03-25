# Automatic Diagnosis for Diabetic Retinopathy

Binary classification of diabetic retinopathy (DR) from color fundus photographs using CNNs in PyTorch.

## Goal

Develop a model that takes a retinal fundus image and outputs a binary diagnosis:

| Label | Meaning |
|-------|---------|
| 0 | No DR |
| 1 | DR (any stage) |

## Repository Structure

```
├── LS5_CV_CNNs_Retinopathy.ipynb   # Main working notebook
├── model/
│   ├── model.py                    # Trainer class and model definitions
│   └── model_components.py         # Reusable network building blocks
├── utils/
│   ├── config.py                   # Hyperparameters and constants
│   └── utils.py                    # Data transforms and preprocessing
├── md/                             # Documentation
│   ├── Lab session description and database.md
│   ├── Design and implementation of the diagnosis system.md
│   ├── Evaluation Metric AUC.md
│   ├── Files.md
│   └── Terms.md
├── data/                           # Dataset — not tracked by git
│   ├── images/                     # 3500 retinal fundus images (.jpg)
│   ├── train.csv                   # 2000 samples
│   ├── val.csv                     # 500 samples
│   └── test.csv                    # 1000 samples (labels unavailable)
├── outputs/                        # Submission CSVs — not tracked by git
├── requirements.txt
└── LICENSE
```

## Documentation

| File | Contents |
|------|----------|
| [Lab session description and database](md/Lab%20session%20description%20and%20database.md) | Problem context, dataset format and splits |
| [Design and implementation of the diagnosis system](md/Design%20and%20implementation%20of%20the%20diagnosis%20system.md) | Pipeline: data loading, custom CNN, fine-tuning |
| [Evaluation Metric AUC](md/Evaluation%20Metric%20AUC.md) | AUC metric, ROC curve, scikit-learn usage |
| [Files](md/Files.md) | Downloads, local folder structure, CSV format |
| [Terms](md/Terms.md) | Evaluation criteria, challenge rules, submission details |

## Setup

```bash
pip install -r requirements.txt
```

Place the extracted dataset under `data/` — see [Files](md/Files.md) for the expected structure.
