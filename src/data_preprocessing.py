# ============================================================
# data_preprocessing.py
# Student Performance Predictor
# Handles: loading, cleaning, scaling, and splitting data
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Features used for prediction
FEATURES = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed']

def load_data(filepath):
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def check_missing_values(df):
    """Check and report missing values in the dataset."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[INFO] No missing values found.")
    else:
        print("[WARNING] Missing values detected:")
        print(missing[missing > 0])
    return missing

def handle_missing_values(df):
    """Fill missing values with column median (robust to outliers)."""
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled missing values in '{col}' with median: {median_val:.2f}")
    return df

def scale_features(X_train, X_test):
    """
    Normalize feature values using StandardScaler.
    Fit only on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[INFO] Features scaled using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler

def preprocess(filepath, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_cls_train, y_cls_test,
             y_reg_train, y_reg_test, scaler
    """
    df = load_data(filepath)
    check_missing_values(df)
    df = handle_missing_values(df)

    X = df[FEATURES]
    y_cls = df['pass']          # classification target (0 or 1)
    y_reg = df['final_score']   # regression target (numeric score)

    # Split into train/test sets
    X_train, X_test, \
    y_cls_train, y_cls_test, \
    y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg,
        test_size=test_size,
        random_state=random_state
    )

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return (X_train_scaled, X_test_scaled,
            y_cls_train, y_cls_test,
            y_reg_train, y_reg_test,
            scaler)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'student_data.csv')
    preprocess(data_path)
    print("[INFO] Preprocessing complete!")
