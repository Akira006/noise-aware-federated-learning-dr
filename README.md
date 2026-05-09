# Noise-Aware Federated Learning for Robust Diabetic Retinopathy Classification

---

## Overview

Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults, affecting over 103 million people globally. Early detection through automated retinal fundus image analysis is critical — yet challenging in low-resource settings where imaging equipment is outdated, lighting is poor, and diagnostic expertise is scarce.

This project proposes a **Noise-Aware Federated Learning** framework for DR classification that improves model robustness under real-world image degradation, without centralizing sensitive patient data. The system simulates a realistic multi-institutional setting by treating three public datasets as independent FL clients with non-IID data distributions.

Three training paradigms are systematically compared across two model architectures, with evaluation on both clean and degraded test sets to quantify robustness.

---

## The Problem

Two core challenges in real-world DR deployment:

1. **Privacy** — Patient data across institutions cannot be freely shared due to regulations such as HIPAA and GDPR
2. **Image Quality Degradation** — Low-resource healthcare facilities produce poor-quality fundus images due to outdated equipment, motion artifacts, and compression — yet most existing FL approaches only simulate degradation via JPEG compression, which fails to reflect real-world complexity

---

## Proposed Solution

**Federated Learning (FedAvg)** enables collaborative model training across institutions without sharing raw data — only model weights are sent to a central server for weighted aggregation.

**Noise-Aware Augmentation Pipeline** simulates realistic image degradation during local training:

| Degradation | Parameter | Simulates |
|-------------|-----------|-----------|
| Gaussian Noise | σ ∈ [0.01, 0.05] | Sensor noise, compression artifacts |
| Gaussian Blur | σ ∈ [1.0, 3.0] | Defocus, motion blur |
| Brightness/Contrast | ±0.2 / [0.7, 1.3] | Low-light, inconsistent acquisition |

Augmentation is applied **stochastically (50% probability per type)** during training — each sample sees a different degradation realization per epoch, encouraging the model to learn robust, degradation-invariant representations.

---

## Experiments

Three paradigms compared:

| # | Paradigm | Description |
|---|----------|-------------|
| 1 | **Centralized** | All data pooled — upper-bound baseline |
| 2 | **Standard FedAvg** | FL without noise augmentation |
| 3 | **Noise-Aware FedAvg** | FL with stochastic noise augmentation *(proposed)* |

Each paradigm is run on two architectures:
- **EfficientNetB0** — primary model, compound scaling efficiency
- **MobileNetV2** — lightweight alternative for resource-constrained environments

---

## Datasets

| Dataset | Images | Description |
|---------|--------|-------------|
| [DDR](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset) | 13,673 | Multi-ethnic, multi-device |
| [EyePACS](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) | 35,126 | Large-scale screening dataset |
| [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | 3,662 | Rural India clinical setting |

All datasets follow the **ICDR grading scale** (5 classes: Grade 0 — No DR to Grade 4 — Proliferative DR). Natural variation across datasets in imaging devices, acquisition conditions, and patient demographics creates realistic **non-IID** distributions across FL clients.

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| FL Communication Rounds | 20 |
| Local Epochs per Round | 5 |
| FL Stage 1 — frozen backbone | Rounds 1–10 |
| FL Stage 2 — unfreeze top-10 layers | Rounds 11–20 |
| Class Weighting | Balanced (inverse-frequency) |
| Data Split | 70:15:15 stratified per client |

Both models use **ImageNet pretrained weights** with two-stage training: backbone frozen first, then top-10 layers unfrozen for domain-specific fine-tuning (BatchNorm remains frozen throughout).

**FL Aggregation (dataset-size-weighted FedAvg):**

$$W_{r+1} = \sum_k \frac{n_k}{n} \cdot W_k^r$$

---

## Evaluation

Each model is evaluated on two test conditions:
- **Clean test set** — unmodified images
- **Noisy test set** — deterministic degradation applied with fixed seed (identical across all 6 notebooks for fair comparison)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision / Recall / F1 | Macro-averaged |
| AUC-ROC | One-vs-Rest macro AUC |
| QWK | Quadratic Weighted Kappa — clinically relevant for ordinal DR grading |
| **Robustness Drop** | `M(clean) − M(noisy)` — lower = more robust |

---

## Repository Structure

```
├── Notebook/
│   ├── MobileNetV2/
│   │   ├── exp1_centralized.ipynb
│   │   ├── exp2_fedavg.ipynb
│   │   └── exp3_noise_aware_fl.ipynb
│   └── EfficientNetB0/
│       ├── exp1_centralized.ipynb
│       ├── exp2_fedavg.ipynb
│       └── exp3_noise_aware_fl.ipynb
└── Result/
    ├── MobileNetV2/
    │   ├── exp1_centralized/
    │   ├── exp2_fedavg/
    │   └── exp3_noise_aware_fl/
    └── EfficientNetB0/
        ├── exp1_centralized/
        ├── exp2_fedavg/
        └── exp3_noise_aware_fl/
```

Each experiment folder contains:
```
├── models/    ← model checkpoints (.keras)
├── logs/      ← metrics JSON, per-class CSV, summary CSV
└── figures/   ← training curves, confusion matrix, ROC, robustness drop
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![Google Colab](https://img.shields.io/badge/Google%20Colab-A100%20GPU-yellow)

---

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
- Mohan Raj et al. (2024). *Federated Learning for Diabetic Retinopathy Diagnosis.* arXiv:2411.00869.
- Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Hendrycks & Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions.* arXiv.

---

## Authors

| Name | Role |
|------|------|
| Akira Agha Nugroho | Conceptualization, Methodology, Software, Data Curation |
| Derry Riccardo | Conceptualization, Methodology, Data Curation, Formal Analysis, Writing |
| Darrell Richie Wibawa | Methodology, Formal Analysis, Writing (original draft & review) |
| Arya Krisna Putra | Supervisor |
| Irene Anindaputri Iswanto | Supervisor |
