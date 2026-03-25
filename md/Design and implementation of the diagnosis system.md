# Practice Guide

This practice provides guidelines to build a baseline reference system through three fundamental steps:

## 1. Database Processing with PyTorch

Build a custom data pipeline to load and preprocess the retinal fundus images using PyTorch's `Dataset` and `DataLoader` abstractions. This includes:

- Reading the CSV files and mapping image ids to file paths
- Handling the eye indicator (mirroring right-eye images)
- Applying data augmentation for training and normalization for evaluation
- Converting the 5-class severity label to a binary label (No DR / DR)

## 2. Custom CNN

Design a feedforward Convolutional Neural Network from scratch that:

- Takes a color fundus image as input
- Extracts visual features through convolutional and pooling layers
- Outputs a binary diagnostic (No DR vs. DR)

This step establishes the **baseline** to compare against more advanced approaches.

## 3. Transfer Learning and Fine-tuning

Take a CNN pretrained on a large-scale general-purpose dataset (e.g. ImageNet) and fine-tune it for the DR diagnostic problem. This involves:

- Loading pretrained weights
- Replacing the classification head for binary output
- Fine-tuning the full network or only the last layers on the DR dataset

---

> The goal of steps 2 and 3 together is to quantify the benefit of transfer learning over training from scratch on a medical imaging dataset of limited size.
