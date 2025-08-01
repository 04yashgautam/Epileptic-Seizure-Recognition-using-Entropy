# 🧠 Epileptic Seizure Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A machine learning pipeline for detecting epileptic seizures from EEG signals using entropy-based features in a **Multilayer Perceptron (MLP)** neural network.

---

## 📋 Table of Contents

1. [📌 Overview](#overview)
2. [✨ Features](#features)
3. [🛠️ Prerequisites](#prerequisites)
4. [⚙️ Installation & Setup](#installation--setup)
5. [📊 Dataset](#dataset)
6. [🔍 Data Preprocessing & Feature Extraction](#data-preprocessing--feature-extraction)
7. [🧪 Modeling](#modeling)
8. [📈 Results](#results)
9. [🧩 Customization & Extensions](#customization--extensions)
10. [🐛 Troubleshooting](#troubleshooting)

---

## 📌 Overview

This project aims to automatically detect **epileptic seizures** from EEG recordings. It extracts **entropy-based features** (Sample Entropy, Approximate Entropy, Spectral Entropy, etc.) and uses a **Multilayer Perceptron (MLP)** neural network to classify seizure vs. non-seizure brain activity.

---

## ✨ Features

✅ **Entropy-Based Features**
  • Approximate Entropy
  • Sample Entropy
  • Spectral Entropy
  • Permutation Entropy

✅ **Robust Validation**
  • k-fold cross-validation for reliability

✅ **Visualizations**
  • ROC curves, Confusion Matrices, Feature Importance

---

## 🛠️ Prerequisites

Make sure you have the following installed:

* Python 3.8+
* NumPy, Pandas, SciPy, Matplotlib
* scikit-learn
* LightGBM
* MNE (optional, for EEG data handling)

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/04yashgautam/epileptic-seizure-recognition.git
cd epileptic-seizure-recognition

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Place your EEG dataset in the 'data/' folder
```

---

## 📊 Dataset

* dataset => https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data

---

## 🔍 Data Preprocessing & Feature Extraction

* **Filtering**: Remove noise and artifacts using bandpass filters.
* **Segmentation**: Split EEG into uniform segments (e.g., 5 seconds).
* **Feature Engineering**: Extract entropy metrics from each segment.

---

## 🧪 Modeling

  * **Multilayer Perceptron (MLP) Neural Network**

* Apply hyperparameter tuning and k-fold cross-validation.

---

## 📈 Results

📊 Evaluation metrics used:

* Accuracy
* Precision, Recall
* F1-score
* ROC-AUC Curve

Visual tools help in comparing classifier performance.

---

## 🧩 Customization & Extensions

🔧 You can extend this project by:

* Adding more advanced features (wavelet transforms, frequency bands, etc.)
* Replacing ML models with deep learning (CNN, LSTM)
* Integrating real-time EEG stream support

---

## 🐛 Troubleshooting

* **Shape Mismatch Errors**
    → Ensure input segment dimensions are consistent across all samples.
* **Overfitting**
    → Use regularization, increase training data, or reduce model complexity.

---

## 📦 requirements.txt

```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
lightgbm>=3.3.0
mne>=1.0.0  # optional, for EEG data handling
joblib>=1.0.0
jupyter>=1.0.0
```

---

## 📫 Contact

Feel free to reach out on [GitHub](https://github.com/04yashgautam) if you have questions or ideas!

---
