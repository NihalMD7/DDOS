# DDoS Attack Detection with Deep Learning

Multi-class classifier that identifies 12 different DDoS attack types from raw network-flow features in the **CICDDoS2019** dataset.

**Stack:** Python · TensorFlow · Keras · PyTorch · Scikit-learn · imbalanced-learn

---

## Overview

Modern DDoS traffic is messy: most flows are benign, a few attack classes dominate the rest, and signature-based detectors miss anything novel. This project trains deep learning models that learn the structure of malicious traffic directly from network-flow features.

Two architectures are cross-validated against the same data pipeline:

1. **Feed-forward neural network** — strong baseline on tabular flow features.
2. **1D Convolutional Neural Network (1D-CNN)** — treats the flow vector as a short sequence and learns local feature interactions.

Both are implemented in **TensorFlow/Keras** and in **PyTorch** for parity, with the same preprocessing and evaluation harness.

## Dataset

- **CICDDoS2019** — Canadian Institute for Cybersecurity, 2019.
- 12 DDoS attack classes plus benign traffic.
- Heavily imbalanced — handled with **SMOTE + random undersampling** to avoid collapsing on the majority class.

## Results

| Improvement                     | Outcome                                  |
| ------------------------------- | ---------------------------------------- |
| Hybrid SMOTE + undersampling    | Restored minority-class recall           |
| GridSearchCV hyperparameter tuning | **+15% validation F1 stability**       |
| TensorFlow ↔ PyTorch parity     | Matching metrics across both frameworks |

## Project Structure

```
ddos-detection/
├── data/             # Place CICDDoS2019 CSVs here (not committed)
├── preprocessing/    # Cleaning, encoding, SMOTE + undersampling
├── models/
│   ├── ffn_tf.py     # Feed-forward in TensorFlow/Keras
│   ├── cnn_tf.py     # 1D-CNN in TensorFlow/Keras
│   └── pytorch/      # PyTorch ports of both architectures
├── notebooks/        # EDA + ablation studies
├── reports/          # Final technical report and plots
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/nihalbabamohammad/ddos-detection
cd ddos-detection
pip install -r requirements.txt

# Download CICDDoS2019 into data/ first.
python preprocessing/build_dataset.py
python models/cnn_tf.py --epochs 25
```

## Reproducing Results

```bash
python models/ffn_tf.py --gridsearch
python models/pytorch/cnn.py --seed 42
```

All training scripts log per-class precision, recall, and F1 to `reports/`.

## Author

Nihal Baba Mohammad — [github.com/nihalbabamohammad](https://github.com/nihalbabamohammad)
