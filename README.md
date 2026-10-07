# Homework 3 — Convolution and Transfer Learning

## Student Information

**Name:** Yousef Aldandani  
**Student ID:** 700783505  
**Course:** CS-5720  

## Overview

This homework covers two topics in convolutional neural networks: implementing 2-D convolution from scratch and comparing frozen feature extraction with fine-tuning using transfer learning.

## Question 1 — Implement Convolution from Scratch

A 2-D convolution operation was implemented using NumPy without using a built-in convolution function. A 3×3 filter was manually moved across a 5×5 input matrix using a stride of 1 and padding of 0.

The resulting output feature map has a shape of 3×3.

## Question 2 — Transfer Learning: Freeze vs. Fine-Tune

A pretrained ResNet18 model was used with the CIFAR-10 image-classification dataset.

Two transfer-learning approaches were compared:

- **Frozen Feature Extractor:** All pretrained layers were frozen, and only the new classification layer was trained.
- **Fine-Tuned Network:** The last convolutional block and the new classification layer were trained.

Both models were trained for 5 epochs.

### Results

| Method | Trainable Parameters | Training Time | Accuracy |
|---|---:|---:|---:|
| Frozen Feature Extractor | 5,130 | 1461.21 s | 80.79% |
| Fine-Tuned Network | 8,398,858 | 1948.62 s | 90.35% |

The fine-tuned model achieved higher accuracy because the unfrozen layers were able to adapt their learned features to the CIFAR-10 dataset. However, it required more training time because it had significantly more trainable parameters.

## Requirements

The notebook uses the following Python libraries:

- NumPy
- PyTorch
- Torchvision
- Matplotlib

## Files

- `HW3.ipynb` — Jupyter Notebook containing the complete code, outputs, plots, and discussion.