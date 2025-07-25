# Epileptic Seizure Recognition Using Entropy-Based Features

This repository implements a machine learning pipeline for detecting epileptic seizures from EEG signals using entropy-based feature extraction and classification algorithms.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation & Setup](#installation--setup)  
5. [Dataset](#dataset)  
6. [Data Preprocessing & Feature Extraction](#data-preprocessing--feature-extraction)  
7. [Modeling](#modeling)  
8. [Results](#results)  
9. [Customization & Extensions](#customization--extensions)  
10. [Troubleshooting](#troubleshooting)  
---

## Overview

This project focuses on automatic detection of epileptic seizures from EEG recordings by extracting entropy measures (e.g., Sample Entropy, Approximate Entropy, Spectral Entropy) as discriminative features and training classifiers such as Random Forest, SVM, and LightGBM.

---

## Features

- **Entropy-Based Features**: Compute Approximate Entropy, Sample Entropy, Spectral Entropy, and Permutation Entropy from EEG signals.  
- **Multiple Classifiers**: Train and compare models including Random Forest, Support Vector Machine (SVM), and LightGBM.  
- **Cross-Validation**: k-fold cross-validation for robust performance estimates.  
- **Visualization**: ROC curves, confusion matrices, and feature importance plots.

---

## Prerequisites

- Python 3.8 or newer  
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib  
- MNE (for EEG data loading, optional)  
- LightGBM  

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/04yashgautam/Epileptic-Seizure-Recognition-using-Entropy.git
   cd Epileptic-Seizure-Recognition-using-Entropy
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Place EEG dataset files in the `data/` directory, following the naming convention used in notebooks.

---

## Dataset

- Public EEG datasets (e.g., Bonn University dataset) containing seizure and non-seizure segments.  

## Data Preprocessing & Feature Extraction

- **Filtering**: Bandpass filter EEG signals to remove artifacts.  
- **Segmentation**: Split continuous EEG into fixed-length windows (e.g., 5 seconds).  
- **Entropy Calculation**: Compute various entropy measures per segment.

---

## Modeling

- Configure model and hyperparameters (e.g., number of trees for Random Forest, kernel choice for SVM).  
- Use cross-validation to select the best-performing classifier.

---

## Results

- Performance metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
  
---

## Customization & Extensions

- **Additional Features**: Integrate other time-domain or frequency-domain features.  
- **Deep Learning Models**: Replace classifiers with CNN/LSTM-based architectures. 

---

## Troubleshooting

- **Data Shape Mismatch**: Ensure consistent window sizes and feature dimensions.  
- **Model Overfitting**: Apply regularization or increase dataset size.
  
---
