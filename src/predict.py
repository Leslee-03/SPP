# ============================================================
# predict.py
# Student Performance Predictor
# Takes user input, loads saved model, outputs prediction
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    """Load saved model bundle from .pkl file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['classifier'], model_data['regressor'], model_data['scaler']

def predict(study_hours, attendance, previous_marks, assignments_completed,
            model_path=None):
    """
    Make a prediction given student features.

    Parameters:
        study_hours           (float) : Hours studied per day
        attendance            (float) : Attendance percentage (0-100)
        previous_marks        (float) : Marks from previous exam (0-100)
        assignments_completed (int)   : Number of assignments completed (0-10)
        model_path            (str)   : Optional path to model.pkl

    Returns:
        dict with 'predicted_score' and 'pass_fail'
    """
    if model_path is None:
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'model', 'model.pkl')

    clf, reg, scaler = load_model(model_path)

    # Prepare input as a DataFrame to match scaler's fitted feature names
    feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed']
    features = pd.DataFrame(
        [[study_hours, attendance, previous_marks, assignments_completed]],
        columns=feature_cols
    )

    # Scale the input using the same scaler used during training
    features_scaled = scaler.transform(features)

    # Make predictions
    predicted_score    = reg.predict(features_scaled)[0]
    predicted_score    = round(float(np.clip(predicted_score, 0, 100)), 2)
    pass_fail_code     = clf.predict(features_scaled)[0]
    pass_fail_label    = "PASS" if pass_fail_code == 1 else "FAIL"
    pass_probability   = clf.predict_proba(features_scaled)[0][1]

    return {
        'predicted_score'   : predicted_score,
        'pass_fail'         : pass_fail_label,
        'pass_probability'  : round(float(pass_probability), 4)
    }

def get_user_input():
    """Prompt user to enter student details via terminal."""
    print("\n" + "="*50)
    print("   STUDENT PERFORMANCE PREDICTOR")
    print("="*50)
    print("Enter student details below:\n")

    study_hours = float(input("  Study Hours per day (e.g. 5.5)  : "))
    attendance  = float(input("  Attendance percentage (e.g. 80)  : "))
    prev_marks  = float(input("  Previous Exam Marks (0-100)      : "))
    assignments = int(input  ("  Assignments Completed (0-10)     : "))

    return study_hours, attendance, prev_marks, assignments

def main():
    study_hours, attendance, prev_marks, assignments = get_user_input()
    result = predict(study_hours, attendance, prev_marks, assignments)

    print("\n" + "="*50)
    print("   PREDICTION RESULTS")
    print("="*50)
    print(f"  Predicted Final Score : {result['predicted_score']:.2f} / 100")
    print(f"  Result                : {result['pass_fail']}")
    print(f"  Pass Probability      : {result['pass_probability']*100:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
