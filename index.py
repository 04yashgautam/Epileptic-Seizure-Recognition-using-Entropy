#!/usr/bin/env python
"""
Entropy-Based Analysis of Brain Activity and Disorder Prediction with User Input
==================================================================================

"Epileptic Seizure Recognition using Entropy" 
dataset => https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data),
performs entropy-based feature extraction on each EEG signal segment, and builds a 
TensorFlow neural network model that classifies the EEG segments into one of several classes.

Author: Yash Gautam
Date: 20th March 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =============================================================================
# ENTROPY-BASED FEATURE FUNCTIONS
# =============================================================================
def shannon_entropy(signal, bins=10):

    counts, bin_edges = np.histogram(signal, bins=bins, density=True)
    probs = counts / np.sum(counts)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def sample_entropy(signal, m=2, r=None):

    if r is None:
        r = 0.2 * np.std(signal)
    N = len(signal)
    
    def _phi(m):
        count = 0.0
        for i in range(N - m):
            template = signal[i:i + m]
            for j in range(i + 1, N - m):
                comparison = signal[j:j + m]
                if np.all(np.abs(template - comparison) < r):
                    count += 1.0
        return count

    B = _phi(m)
    A = _phi(m + 1)
    # Avoid division by zero
    if B == 0:
        return np.inf
    return -np.log(A / B)

def spectral_entropy(signal, fs=256):

    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals[:len(fft_vals) // 2])
    psd = fft_vals ** 2
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0
    psd_norm = psd / psd_sum
    entropy = -np.sum([p * np.log2(p + 1e-12) for p in psd_norm])
    return entropy

# =============================================================================
# FEATURE EXTRACTION FUNCTION
# =============================================================================
def extract_features(signal_row):

    signal_row = np.array(signal_row, dtype=np.float64)
    mean_val = np.mean(signal_row)
    std_val = np.std(signal_row)
    var_val = np.var(signal_row)
    shannon_val = shannon_entropy(signal_row, bins=20)
    sample_val = sample_entropy(signal_row, m=2)
    spectral_val = spectral_entropy(signal_row, fs=256)
    features = [mean_val, std_val, var_val, shannon_val, sample_val, spectral_val]
    return features

# =============================================================================
# DATA LOADING AND PREPARATION FUNCTIONS
# =============================================================================
def load_data(file_path):

    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}.")
    except Exception as e:
        print(f"Error reading dataset from {file_path}: {e}")
        sys.exit(1)
        
    # Remove any unnamed index columns if present
    for col in data.columns:
        if "Unnamed" in col:
            data = data.drop(columns=col)
            print(f"Dropped column: {col} (considered an index column).")
    
    print("\nPreview of loaded data:")
    print(data.head())
    return data

def prepare_dataset(data):

    columns = data.columns.tolist()
    label_col = columns[-1]
    signal_cols = columns[:-1]

    X_raw = data[signal_cols].values
    y_raw = data[label_col].values

    print("\nInitial feature matrix shape:", X_raw.shape)
    print("Label vector shape:", y_raw.shape)

    features_list = []
    total_samples = X_raw.shape[0]
    print("\nExtracting features from each EEG sample...")
    start_time = time.time()
    for i in range(total_samples):
        features = extract_features(X_raw[i])
        features_list.append(features)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total_samples} samples.")
    end_time = time.time()
    print(f"Completed feature extraction in {end_time - start_time:.2f} seconds.")

    X_features = np.array(features_list)
    print("Feature matrix shape:", X_features.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    print("Data standardized.")

    y_int = y_raw.astype(np.int32)
    y_int = y_int - 1  # Adjusting to zero-based index
    y_encoded = tf.keras.utils.to_categorical(y_int, num_classes=5)
    print("Labels one-hot encoded with shape:", y_encoded.shape)

    return X_scaled, y_encoded, scaler

# =============================================================================
# VISUALIZATION FUNCTION FOR ENTROPY DISTRIBUTION
# =============================================================================
def visualize_entropy_distribution(X, scaler):

    X_inv = scaler.inverse_transform(X)
    shannon_vals = X_inv[:, 3]
    sample_vals = X_inv[:, 4]
    spectral_vals = X_inv[:, 5]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(shannon_vals, bins=30, color='skyblue', edgecolor='black')
    plt.title("Shannon Entropy Distribution")
    plt.xlabel("Shannon Entropy")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(sample_vals, bins=30, color='salmon', edgecolor='black')
    plt.title("Sample Entropy Distribution")
    plt.xlabel("Sample Entropy")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 3)
    plt.hist(spectral_vals, bins=30, color='limegreen', edgecolor='black')
    plt.title("Spectral Entropy Distribution")
    plt.xlabel("Spectral Entropy")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# =============================================================================
# TENSORFLOW MODEL BUILDING FUNCTION
# =============================================================================
def build_model(input_dim, num_classes):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("\nModel built and compiled. Summary:")
    model.summary(print_fn=lambda x: print(x))
    return model

# =============================================================================
# TRAINING HISTORY PLOTTING
# =============================================================================
def plot_training_history(history):

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], marker='o', label='Training Loss')
    plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# CONFUSION MATRIX PLOTTING FUNCTION
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, classes):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Confusion Matrix")
    plt.show()

# =============================================================================
# SCALER SAVE/LOAD FUNCTIONS
# =============================================================================
def save_scaler(scaler, filename='scaler.pkl'):

    import joblib
    joblib.dump(scaler, filename)
    print(f"Scaler saved to '{filename}'.")

def load_scaler(filename='scaler.pkl'):

    import joblib
    scaler = joblib.load(filename)
    print(f"Scaler loaded from '{filename}'.")
    return scaler

# =============================================================================
# USER INPUT PREDICTION FUNCTION
# =============================================================================
def user_input_prediction(model, scaler):
    
    print("\n------ Custom EEG Signal Prediction ------")
    while True:
        response = input("\nWould you like to enter a custom EEG signal for prediction? (y/n): ").strip().lower()
        if response == 'y':
            try:
                user_input_data = input("Enter EEG signal values separated by commas:\n")
                signal = [float(val.strip()) for val in user_input_data.split(',')]
                features = extract_features(signal)
                features = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features)
                pred_prob = model.predict(features_scaled)
                pred_class = np.argmax(pred_prob, axis=1)[0]
                print(f"Predicted Class: Class {pred_class + 1}")
            except Exception as e:
                print("Error processing input or generating prediction:", e)
        elif response == 'n':
            print("Exiting custom prediction mode.")
            break
        else:
            print("Invalid input. Please type 'y' or 'n'.")

# =============================================================================
# MAIN FUNCTION: EXECUTION
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATA
    # -------------------------------------------------------------------------
    data_file = "epileptic_seizure_data.csv"  # Modify this path as necessary.
    data = load_data(data_file)

    # -------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET (FEATURE EXTRACTION & LABEL ENCODING)
    # -------------------------------------------------------------------------
    X, y, scaler = prepare_dataset(data)

    # -------------------------------------------------------------------------
    # STEP 3: VISUALIZE ENTROPY DISTRIBUTION
    # -------------------------------------------------------------------------
    print("\nVisualizing entropy feature distributions...")
    visualize_entropy_distribution(X, scaler)

    # -------------------------------------------------------------------------
    # STEP 4: SPLIT DATA INTO TRAINING AND TEST SETS
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nSplit completed: {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")

    # -------------------------------------------------------------------------
    # STEP 5: BUILD THE TENSORFLOW MODEL
    # -------------------------------------------------------------------------
    input_dim = X_train.shape[1]
    num_classes = y.shape[1]
    model = build_model(input_dim, num_classes)

    # -------------------------------------------------------------------------
    # STEP 6: TRAIN THE MODEL
    # -------------------------------------------------------------------------
    print("\nCommencing model training...")
    start_training = time.time()
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)
    end_training = time.time()
    print(f"Training finished in {(end_training - start_training):.2f} seconds.")

    # -------------------------------------------------------------------------
    # STEP 7: EVALUATE THE MODEL ON TEST DATA
    # -------------------------------------------------------------------------
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # -------------------------------------------------------------------------
    # STEP 8: VISUALIZE TRAINING HISTORY
    # -------------------------------------------------------------------------
    plot_training_history(history)

    # -------------------------------------------------------------------------
    # STEP 9: PERFORMANCE METRICS & CONFUSION MATRIX
    # -------------------------------------------------------------------------
    print("\nGenerating predictions on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred,
                                   target_names=[f"Class {i+1}" for i in range(num_classes)])
    print(report)

    plot_confusion_matrix(y_true, y_pred, classes=[f"Class {i+1}" for i in range(num_classes)])

    # -------------------------------------------------------------------------
    # STEP 10: SAVE MODEL AND SCALER
    # -------------------------------------------------------------------------
    model.save("epileptic_seizure_model.h5")
    print("Model saved as 'epileptic_seizure_model.h5'.")
    save_scaler(scaler, filename="scaler.pkl")

    # -------------------------------------------------------------------------
    # STEP 11: INTERACTIVE USER INPUT FOR CUSTOM PREDICTION
    # -------------------------------------------------------------------------
    user_input_prediction(model, scaler)

    print("\nExecution completed successfully.")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
