# Lab Session Description and Database

## Problem Description

Diabetic Retinopathy (DR) is the leading cause of blindness in the working-age population of the developed world. World Health Organization estimates that 347 million people have the disease worldwide. DR is an eye disease associated with long-standing diabetes. Around 40% to 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.

Our goal is to develop a CNN providing an automatic diagnosis of DR with color fundus photography as input. The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning.

## Input Data

A large set of high-resolution retina images taken under a variety of imaging conditions. Left and right fields can be provided indistinctively. Each image is described by:

1. An image id.
2. An eye indicator (`left` / `right`).
3. A label indicating the presence of diabetic retinopathy on a scale of 0 to 4:

| Label | Severity |
|-------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Dataset

The dataset contains **3500 images** divided into 3 splits:

| Split | Images |
|-------|--------|
| Training | 2000 |
| Validation | 500 |
| Test | 1000 |

### CSV Format

Each split has an associated CSV file where each line corresponds to a clinical case with the following fields (comma-separated):

1. **Image id** — numerical id used to build the path to the image.
2. **Eye indicator** — `0` for left eye, `1` for right eye. Allows mirroring right-eye images so that both eyes are comparable.
3. **Label** — integer from 0 to 4 (available for training and validation only). The test set uses `-1` as a placeholder since labels are not available.

## Task

For simplicity, the original 5-class labels are transformed into a **binary label**:

| Binary Label | Meaning |
|-------------|---------|
| 0 | No DR |
| 1 | DR (any stage) |

This reduces the problem to a **binary classification** task.

Students may use the training and validation sets to build their solutions and must provide the predicted scores associated with the test set.
