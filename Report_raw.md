# Project Report — Automatic Diagnosis of Diabetic Retinopathy

> **Limit: 2 A4 sides.** Side 1 = main text. Side 2 = tables, figures and references.

---

## Individual Contributions

Both members contributed jointly to the overall system design, experimental decisions, and final integration. Responsibilities were divided by track to parallelise development, but all major choices were discussed and validated together.

| Member | Primary Track | Key Contributions |
|--------|--------------|-------------------|
| Jorge Barcenilla | Model 1 (Custom) | SEResNet9 architecture, BenGraham + PerImageNormalize + RandomCutout preprocessing pipeline, FocalLoss + WeightedRandomSampler + SequentialLR warmup, modular `src/` codebase and test suite. |
| Santiago Prieto | Model 2 (Fine-Tuning) | DualChannelEnhancement (dual-channel CLAHE), EfficientNet-B2 + DenseNet-121 multi-scale ensemble with learnable Softmax weights, FocalLoss + WeightedRandomSampler, TTA (10 passes), modular `src/` and `utils/` structure. |

---

# Side 1 — System Description

## Model 1: Custom Architecture from Scratch (SEResNet9)

### Architecture

SEResNet9 is a ResNet with **8 convolutional layers** (~1.65 M trainable parameters) and Squeeze-and-Excitation (SE) blocks inserted after each ResBlock. Although the model is conventionally named ResNet9 (following the DAWNBench naming tradition), the actual Conv2d count is 8: one stem layer, one layer per downsampling stage (×3), and two convolutions per ResBlock (×2). The SE blocks apply channel-wise attention via global-average-pooling → FC → ReLU → FC → Sigmoid, suppressing irrelevant channels and amplifying informative ones — in particular the green channel, where microaneurysms and exudates exhibit the highest contrast — at negligible computational cost. Dropout(0.1) and Kaiming initialisation are used for gradient stability when training from random weights.

### Preprocessing

The pipeline enforces a critical ordering. `CropByEye` is applied **first**, removing black borders via brightness thresholding before any other transformation. `BenGraham` (σ=15) is then applied: `cv2.addWeighted(img, 4, GaussBlur(img, σ=15), −4, 128)` maps the background to neutral grey (128) and maximises the local contrast of lesions. The order is invariant: if BenGraham precedes CropByEye, black borders are converted to uniform grey and the brightness-based eye detector fails. Finally, `PerImageNormalize` standardises each image with its own per-channel statistics, eliminating the domain bias introduced by ImageNet normalisation.

### Data Augmentation

`RandomHorizontalFlip(p=0.5)`, `RandomRotation(180°)` and `ColorJitter` with **hue=0** (hue jitter is omitted because colour encodes lesion type). The differential contribution is `RandomCutout`: two 32×32 px patches filled with value 0.5 force the network to learn distributed representations across the whole retina, preventing reliance on localised regions and improving generalisation to peripheral lesions.

### Class Imbalance and Training

Class imbalance is addressed with `WeightedRandomSampler` (50/50 balanced batches) combined with `FocalLoss(γ=2.0, α=0.5)`. The symmetric α=0.5 deliberately avoids double-correction, since the sampler already balances classes at the batch level. The `SequentialLR` scheduler chains a linear warmup (epochs 0–4, initial factor 0.01) with `CosineAnnealingLR` (epochs 5–59, up to a maximum of 60 epochs): warmup prevents collapse toward the majority class during random initialisation. **Early stopping** with patience 15 terminates training if validation AUC does not improve for 15 consecutive epochs; epoch 53 was the best.

**Result:** val AUC = **0.7292** (epoch 53) · Codabench AUC = **0.74**.

---

## Model 2: Fine-Tuning with Multi-Scale Heterogeneous Ensemble

### Preprocessing

After `CropByEye` and resizing with margin, `DualChannelEnhancement` is applied: CLAHE (clip=2.0, grid 4×4) independently on the green and red channels, with a fundus mask to exclude black borders from amplification. The justification is medical and supported by the literature: Sahlsten et al. (*Diabetic Retinopathy Screening Using Machine Learning: A Systematic Review*, 2019) identify the green channel as the most informative for detecting microaneurysms and hard exudates in fundus photographs; the red channel complements by highlighting haemorrhages and neovascularisation. Adaptive CLAHE avoids the saturation of global histogram equalisation in bright optic disc regions.

### Data Augmentation

Geometric transforms (flips, anatomically plausible rotations of ±30°) and `ColorJitter` — with a small `hue=0.05` to simulate inter-camera colour variability across fundus imaging devices, unlike the Custom track where hue is set to 0 — are combined with `GaussianNoise(σ=0.02)`. Gaussian noise further regularises against sensor variability specific to the medical domain.

### Architecture — Multi-Scale Heterogeneous Ensemble

The ensemble combines **EfficientNet-B2** and **DenseNet-121**, pretrained on ImageNet, with the **last three layer groups of each backbone unfrozen** at initialisation while all earlier layers remain frozen throughout training. Transfer learning is essential given the scarcity of labelled medical data: low-level filters learned on ImageNet transfer effectively and accelerate convergence. The key technique is **multi-scale input**: EfficientNet-B2 receives the image interpolated to 224 px (global context), DenseNet-121 receives it at 512 px (fine lesion detail), promoting feature complementarity between the two architectures. **Ensemble weights are learnable parameters** (`nn.Parameter` normalised with Softmax): the network dynamically learns which architecture and scale are more reliable for the final prediction.

### Class Imbalance and Training

As in the Custom track, `WeightedRandomSampler` (50/50 batches) and `FocalLoss(γ=2.0, α=0.25)` are used as a dual strategy against severe class imbalance. The lower α=0.25 — compared to 0.5 in the Custom track — provides a more aggressive penalty on the positive class on top of the already-balanced batches from the sampler. AdamW optimiser with `ReduceLROnPlateau` (mode `max`, factor=0.3, patience=3) and Early Stopping (patience 10) monitoring validation AUC ROC.

### Inference — Test-Time Augmentation (TTA)

10 stochastic passes with random flips and rotations over each test image; the output probabilities are averaged. TTA reduces prediction variance on unseen data, providing statistical robustness equivalent to a temporal micro-ensemble over the same image.

### Results — Per-Model and Ensemble Performance

| Model | Resolution | Val AUC |
|-------|:----------:|:-------:|
| EfficientNet-B2 | 224 px | 0.6044 |
| EfficientNet-B2 | 384 px | 0.6712 |
| EfficientNet-B2 | 512 px | 0.6385 |
| DenseNet-121 | 224 px | 0.7316 |
| DenseNet-121 | 384 px | 0.8011 |
| DenseNet-121 | 512 px | 0.8095 |
| **Full ensemble** | 224 / 512 px | **0.8413** |

The ensemble outperforms all individual models, demonstrating the benefit of combining heterogeneous architectures and multi-scale inputs. **Codabench AUC = 0.8189**.

---

# Side 2 — Supplementary Material

## Iteration Process and Experiments

### Custom Track — Chronological Evolution

| # | Experiment | Decision / result |
|---|------------|------------------|
| 1 | **VGG-inspired baseline** (BCEWithLogitsLoss, Adam, 224 px) | Starting point; slow convergence, moderate AUC |
| 2 | **EnsembleNet: DenseNetSmall (green channel, 82k params) + ResNet-9 (RGB, 610k params)** | Channel split did not yield sufficient gain for its training cost |
| 3 | **Scale to 512 px + AMP (mixed precision) + CosineAnnealingLR** | Better resolution for fine lesions; AMP allows larger batch without extra VRAM |
| 4 | **BenGraham (σ=15) + PerImageNormalize** | Improved lesion contrast; per-image normalisation removes ImageNet domain bias |
| 5 | **CropByEye → BenGraham ordering (bug fix)** | Reversed order caused BenGraham to convert black borders to grey, breaking the eye detector |
| 6 | **FocalLoss (γ=2.0)** | Broke the loss plateau caused by easy-example gradient dominance |
| 7 | **SEResNet9 with SE blocks** | Channel attention at 1.65 M params: better capacity/cost ratio than EnsembleNet |
| 8 | **WeightedRandomSampler + linear warmup 5 epochs** | Sampler ensures balanced batches; warmup prevents initial collapse toward the majority class |

### Fine-Tuning Track — Chronological Evolution

| # | Experiment | Decision / result |
|---|------------|------------------|
| 1 | **EfficientNet-B2 baseline fine-tuned** | First submission: Codabench AUC ~0.75 |
| 2 | **Ensemble analysis** | Study of which architectures and joint weights maximise AUC |
| 3 | **Combined version: multi-backbone + `main_combined.ipynb`** | Exploration of heterogeneous combinations (EfficientNet + DenseNet) with uniform voting |
| 4 | **WIP multi-scale ensemble** | Prototype of multi-resolution interpolation flow per backbone |
| 5 | **Multi-resolution ensemble (EfficientNet-B2 @ 224 px + DenseNet-121 @ 512 px)** | Final design: learnable Softmax weights replace uniform averaging; DualChannelEnhancement integrated |
| 6 | **TTA (10 passes) at inference** | Reduces test variance; net positive gain without retraining cost |

---

## Results Table

| Metric | Custom (SEResNet9) | Fine-Tuning Ensemble |
|--------|:-----------------:|:-------------------:|
| Val AUC (best, local) | 0.7292 (ep. 53) | 0.8413 |
| **Codabench AUC (public test)** | **0.74** | **0.8189** |
| Trainable parameters | ~1.65 M | ~14 M (EfficientNet-B2 + DenseNet-121) |
| Training epochs (max) | 60 (early stop p=15) | 50 (early stop p=10) |
| Input resolution | 512×512 | 224 px / 512 px (multi-scale) |
| Loss | FocalLoss (α=0.5, γ=2.0) | FocalLoss (α=0.25, γ=2.0) |
| LR scheduler | SequentialLR (linear warmup + CosineAnnealing) | ReduceLROnPlateau |
| TTA at test | No | Yes (10 passes) |
| Pretrained weights | No (from scratch) | ImageNet |

## Proposed Figures

1. **Figure 1 — ROC curves (both models)**: validation ROC curves of SEResNet9 (Custom, AUC=0.74) and Multi-Scale Ensemble (Fine-Tune, AUC=0.8189) overlaid on the same plot, with the random baseline (AUC=0.50) as reference. Directly compares the discriminative capacity of both approaches.

2. **Figure 2 — Prediction score distribution by class**: histogram of predicted probabilities on the validation set, split by ground-truth class (No DR / DR), for each model. Shows the degree of class separation and output score calibration of both systems.

3. **Figure 3 — Visual example of transforms**: image grid showing, for the same fundus photograph: (a) original image, (b) after `CropByEye` + `BenGraham` + `PerImageNormalize` (Custom), (c) after `CropByEye` + `DualChannelEnhancement` CLAHE (Fine-Tune), (d) example of `RandomCutout` applied to the preprocessed Custom-track image.

## References

| Technique | Source |
|-----------|--------|
| SE blocks (Squeeze-and-Excitation) | Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 |
| BenGraham preprocessing | Graham, *Kaggle Diabetic Retinopathy Detection*, Kaggle 2015 |
| Focal Loss | Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet), ICCV 2017 |
| EfficientNet | Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019 |
| DenseNet | Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017 |
| CLAHE | Zuiderveld, *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV, 1994 |
| Green channel in DR fundus images | Sahlsten et al., *Diabetic Retinopathy Screening Using Machine Learning: A Systematic Review*, J. Diabetes Res. 2019 |
| Test-Time Augmentation | Shanmugam et al., *Better Aggregation in Test-Time Augmentation*, ICCV 2021 |
| Class imbalance + Focal Loss | Buda et al., *A systematic study of the class imbalance problem in CNNs*, Neural Networks 2018 |
| ResNet | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 |
