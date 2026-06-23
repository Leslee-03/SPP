# ============================================================
# app.py — Streamlit Web Application
# Student Performance Predictor
# Run with: streamlit run app/app.py
# ============================================================

import streamlit as st
import sys
import os

# Allow import of predict.py from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict

# --- Page Config ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- Header ---
st.title("🎓 Student Performance Predictor")
st.markdown(
    "Predict whether a student will **Pass or Fail** and estimate their "
    "**Final Score** using machine learning."
)
st.markdown("---")

# --- Input Form ---
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "📚 Study Hours per Day",
        min_value=0.0, max_value=12.0, value=5.0, step=0.5,
        help="Average hours a student studies per day"
    )
    attendance = st.slider(
        "🏫 Attendance (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Percentage of classes attended"
    )

with col2:
    previous_marks = st.slider(
        "📝 Previous Exam Marks",
        min_value=0.0, max_value=100.0, value=65.0, step=1.0,
        help="Marks obtained in the previous exam (out of 100)"
    )
    assignments_completed = st.slider(
        "✅ Assignments Completed",
        min_value=0, max_value=10, value=7, step=1,
        help="Number of assignments submitted (out of 10)"
    )

st.markdown("---")

# --- Predict Button ---
if st.button("🔮 Predict Performance", use_container_width=True):
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'model', 'model.pkl'
    )

    if not os.path.exists(model_path):
        st.error(
            "❌ Model not found! Please run `python src/train_model.py` first."
        )
    else:
        result = predict(
            study_hours, attendance,
            previous_marks, assignments_completed,
            model_path=model_path
        )

        # Display results
        st.subheader("📊 Prediction Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                label="🎯 Predicted Score",
                value=f"{result['predicted_score']:.1f} / 100"
            )

        with r2:
            label = "✅ PASS" if "PASS" in result['pass_fail'] else "❌ FAIL"
            st.metric(label="📋 Result", value=label)

        with r3:
            st.metric(
                label="📈 Pass Probability",
                value=f"{result['pass_probability']*100:.1f}%"
            )

        # Progress bar for score
        st.markdown("#### Score Meter")
        st.progress(min(100, max(0, int(result['predicted_score']))))

        # Feedback message
        score = result['predicted_score']
        if score >= 85:
            st.success("🌟 Excellent performance! Keep it up!")
        elif score >= 70:
            st.info("👍 Good performance. A little more effort will make it great!")
        elif score >= 50:
            st.warning("⚠️ Borderline. Focus more on studies and attendance.")
        else:
            st.error("❗ At risk of failing. Immediate improvement needed.")

# --- Footer ---
st.markdown("---")
st.caption("Soft Computing Student Performance Predictor Project")
