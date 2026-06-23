# ============================================================
# train_model.py
# Student Performance Predictor
# Trains: Logistic Regression (classification) +
#         Linear Regression (regression)
# Saves: model.pkl
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, r2_score,
                             mean_absolute_error, mean_squared_error)
import sys

# Add parent directory to path so we can import data_preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess

def train_classifier(X_train, y_train):
    """Train Logistic Regression for Pass/Fail classification."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("[INFO] Logistic Regression model trained.")
    return clf

def train_regressor(X_train, y_train):
    """Train Linear Regression for final score prediction."""
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print("[INFO] Linear Regression model trained.")
    return reg

def evaluate_classifier(model, X_test, y_test):
    """Evaluate classification model and print metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print("\n" + "="*50)
    print("  CLASSIFICATION RESULTS (Pass/Fail)")
    print("="*50)
    print(f"  Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Fail', 'Pass'],
                                zero_division=0))
    return acc, cm

def evaluate_regressor(model, X_test, y_test):
    """Evaluate regression model and print metrics."""
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\n" + "="*50)
    print("  REGRESSION RESULTS (Final Score)")
    print("="*50)
    print(f"  R² Score        : {r2:.4f}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  RMSE            : {rmse:.4f}")
    print("="*50)
    return r2, mae, rmse

def save_model(clf, reg, scaler, model_dir):
    """Save trained models and scaler to a .pkl file."""
    os.makedirs(model_dir, exist_ok=True)
    model_data = {
        'classifier': clf,
        'regressor': reg,
        'scaler': scaler
    }
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[INFO] Model saved to: {model_path}")
    return model_path

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(base_dir, 'data', 'student_data.csv')
    model_dir  = os.path.join(base_dir, 'model')

    # Step 1: Preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    (X_train, X_test,
     y_cls_train, y_cls_test,
     y_reg_train, y_reg_test,
     scaler) = preprocess(data_path)

    # Step 2: Train models
    print("\n[STEP 2] Training models...")
    clf = train_classifier(X_train, y_cls_train)
    reg = train_regressor(X_train, y_reg_train)

    # Step 3: Evaluate models
    print("\n[STEP 3] Evaluating models...")
    acc, cm = evaluate_classifier(clf, X_test, y_cls_test)
    r2, mae, rmse = evaluate_regressor(reg, X_test, y_reg_test)

    # Step 4: Save models
    print("\n[STEP 4] Saving models...")
    save_model(clf, reg, scaler, model_dir)

    print("\n[SUCCESS] Training pipeline complete!")
    print(f"  Classification Accuracy : {acc*100:.2f}%")
    print(f"  Regression R² Score     : {r2:.4f}")

if __name__ == "__main__":
    main()
