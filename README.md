# Handwritten Signature Verification

A deep learning system for verifying the authenticity of handwritten signatures using a **CNN embedding + XGBoost** pipeline, trained and evaluated on the CEDAR benchmark dataset.

---

## Overview

This project addresses the problem of **offline signature verification** — determining whether a given signature is genuine or a skilled forgery. Rather than classifying signatures directly, the model learns a compact feature embedding via CNN, then uses classical ML classifiers to make the final decision.

**Pipeline:**
```
Input Image → Preprocessing → CNN Embedding Extractor → Feature Difference → XGBoost / SVM → Genuine / Forged
```

---

## Dataset — CEDAR

| Property | Value |
|---|---|
| Persons | 55 |
| Genuine signatures / person | 24 |
| Forged signatures / person | 24 |
| Total images | 2,640 |
| Image format | PNG (grayscale) |

---

## Preprocessing

Each signature image goes through a 5-step pipeline before being fed into the model:

1. **Grayscale conversion** — read as single-channel
2. **Resize** to 155 × 220 px (bilinear interpolation)
3. **Gaussian blur** (3×3 kernel) — noise reduction
4. **Otsu thresholding** — binarization
5. **Centering** — shift signature to image center

---

## Model Architecture

### CNN Embedding Extractor
A compact convolutional network that maps a signature image to a fixed-size feature vector (embedding).

### Classifiers
Two classifiers are trained on the **element-wise absolute difference** between a query embedding and a reference signature profile:

| Classifier | Role |
|---|---|
| **XGBoost** | Primary classifier |
| **SVM** | Secondary / comparison |

---

## Tech Stack

- **PyTorch** — CNN model
- **OpenCV** — image preprocessing
- **XGBoost** — gradient boosted classifier
- **scikit-learn** — SVM, evaluation metrics
- **Matplotlib** — visualization

---

## Project Structure

```
signature-verification/
├── signature_verification.ipynb   # Main notebook (end-to-end pipeline)
└── README.md
```

---

## Usage

The notebook is designed to run in **Google Colab**:

1. Mount Google Drive and set `DATA_DIR` to your CEDAR dataset path
2. Run all cells sequentially (Cells 1–20) to train the model
3. Use **Cell 21–22** to build a personal signature profile
4. Use **Cell 23–25** to verify individual signatures interactively
5. Use **Cell 26** for batch verification from a Google Drive folder

---

## Results

Evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC) are computed on a held-out test split. Confusion matrices and ROC curves are plotted inline in the notebook.

---

## Skills Demonstrated

- Custom CNN architecture design in PyTorch
- Classical ML integration (XGBoost, SVM) with deep features
- Image preprocessing pipeline (OpenCV)
- Metric-based evaluation (precision/recall/F1/AUC)
- End-to-end deployment in an interactive Colab notebook
