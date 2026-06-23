# ============================================================
# generate_visualizations.py
# Student Performance Predictor — Full Visualization Suite
# Generates and saves all plots to ../reports/
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error,
    ConfusionMatrixDisplay
)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'student_data.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi'      : 150,
    'font.family'     : 'DejaVu Sans',
    'axes.titlesize'  : 14,
    'axes.titleweight': 'bold',
    'axes.labelsize'  : 12,
})

PASS_COLOR = '#2ECC71'
FAIL_COLOR = '#E74C3C'
BLUE       = '#2980B9'
DARK_BLUE  = '#1A252F'

def savefig(name):
    path = os.path.join(REPORTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] {name}")
    return path

# ── Load & prepare data ──────────────────────────────────────
print("\n[1/9] Loading data...")
df = pd.read_csv(DATA_PATH)
FEATURES = ['study_hours', 'attendance', 'previous_marks', 'assignments_completed']
X = df[FEATURES]
y_cls = df['pass']
y_reg = df['final_score']

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_cls_train)

reg = LinearRegression()
reg.fit(X_train_sc, y_reg_train)

y_pred_cls   = clf.predict(X_test_sc)
y_prob       = clf.predict_proba(X_test_sc)[:, 1]
y_pred_score = reg.predict(X_test_sc)

# ════════════════════════════════════════════════════════════
# PLOT 1 — Score Distribution + Pass/Fail Pie
# ════════════════════════════════════════════════════════════
print("[PLOT 1] Score Distribution & Pass/Fail Pie...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['final_score'], bins=20, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].axvline(df['final_score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['final_score'].mean():.1f}")
axes[0].axvline(df['final_score'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {df['final_score'].median():.1f}")
axes[0].set_title('Distribution of Final Scores')
axes[0].set_xlabel('Final Score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Pie
counts = df['pass'].value_counts().sort_index()
labels = [f'Fail\n(n={counts.get(0,0)})', f'Pass\n(n={counts.get(1,0)})']
axes[1].pie(counts, labels=labels, autopct='%1.1f%%',
            colors=[FAIL_COLOR, PASS_COLOR], startangle=90,
            explode=(0.05, 0.05), shadow=True,
            textprops={'fontsize': 12})
axes[1].set_title('Pass vs Fail Distribution')

plt.tight_layout()
savefig('01_score_distribution.png')

# ════════════════════════════════════════════════════════════
# PLOT 2 — Study Hours vs Final Score Scatter
# ════════════════════════════════════════════════════════════
print("[PLOT 2] Study Hours vs Final Score...")
fig, ax = plt.subplots(figsize=(10, 6))
colors = df['pass'].map({1: PASS_COLOR, 0: FAIL_COLOR})
ax.scatter(df['study_hours'], df['final_score'],
           c=colors, alpha=0.7, edgecolors='white', s=80)
m, b = np.polyfit(df['study_hours'], df['final_score'], 1)
x_line = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 100)
ax.plot(x_line, m * x_line + b, color=DARK_BLUE, linewidth=2.5,
        label=f'Trend (slope={m:.2f})')
pass_patch = mpatches.Patch(color=PASS_COLOR, label='Pass')
fail_patch = mpatches.Patch(color=FAIL_COLOR, label='Fail')
ax.legend(handles=[pass_patch, fail_patch,
                   plt.Line2D([0],[0], color=DARK_BLUE, linewidth=2, label='Trend')])
ax.set_xlabel('Study Hours per Day')
ax.set_ylabel('Final Score')
ax.set_title('Study Hours vs Final Score')
plt.tight_layout()
savefig('02_study_hours_vs_score.png')

# ════════════════════════════════════════════════════════════
# PLOT 3 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
print("[PLOT 3] Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, vmin=-1, vmax=1,
            linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
savefig('03_correlation_heatmap.png')

# ════════════════════════════════════════════════════════════
# PLOT 4 — Feature Importance (|Coefficients|)
# ════════════════════════════════════════════════════════════
print("[PLOT 4] Feature Importance...")
feat_labels = ['Study Hours', 'Attendance', 'Previous Marks', 'Assignments']
importances = pd.Series(np.abs(reg.coef_), index=feat_labels).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importances.index, importances.values,
               color=['#AED6F1','#5DADE2','#2E86C1','#1A5276'],
               edgecolor='white', height=0.55)
for bar, val in zip(bars, importances.values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=11)
ax.set_xlabel('|Coefficient| (Relative Importance)')
ax.set_title('Feature Importance — Linear Regression Coefficients')
ax.set_xlim(0, importances.max() * 1.2)
plt.tight_layout()
savefig('04_feature_importance.png')

# ════════════════════════════════════════════════════════════
# PLOT 5 — Box Plots by Pass/Fail
# ════════════════════════════════════════════════════════════
print("[PLOT 5] Box Plots by Pass/Fail...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
feat_map = {'study_hours':'Study Hours / Day',
            'attendance':'Attendance (%)',
            'previous_marks':'Previous Exam Marks',
            'assignments_completed':'Assignments Completed'}
for ax, (col, label) in zip(axes.flatten(), feat_map.items()):
    data_fail = df[df['pass']==0][col]
    data_pass = df[df['pass']==1][col]
    bp = ax.boxplot([data_fail, data_pass], labels=['Fail','Pass'],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor(FAIL_COLOR)
    bp['boxes'][1].set_facecolor(PASS_COLOR)
    ax.set_title(f'{label} by Outcome')
    ax.set_ylabel(label)
plt.suptitle('Feature Distribution by Pass/Fail Outcome',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
savefig('05_boxplots_by_outcome.png')

# ════════════════════════════════════════════════════════════
# PLOT 6 — Confusion Matrix
# ════════════════════════════════════════════════════════════
print("[PLOT 6] Confusion Matrix...")
cm = confusion_matrix(y_cls_test, y_pred_cls)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail','Pass'])
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')

# Annotate cells with extra info
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / cm.sum() * 100
        ax.text(j, i + 0.3, f'({pct:.1f}%)', ha='center', va='center',
                fontsize=10, color='white' if val > cm.max()/2 else 'gray')

cell_labels = {(0,0):'True\nNegative', (0,1):'False\nPositive',
               (1,0):'False\nNegative', (1,1):'True\nPositive'}
for (r,c), lbl in cell_labels.items():
    ax.text(c, r - 0.3, lbl, ha='center', va='center',
            fontsize=9, color='white' if cm[r,c] > cm.max()/2 else 'gray',
            style='italic')

acc = accuracy_score(y_cls_test, y_pred_cls)
ax.set_title(f'Confusion Matrix  |  Accuracy = {acc*100:.1f}%')
plt.tight_layout()
savefig('06_confusion_matrix.png')

# ════════════════════════════════════════════════════════════
# PLOT 7 — ROC Curve
# ════════════════════════════════════════════════════════════
print("[PLOT 7] ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_cls_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#2980B9', lw=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0,1],[0,1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.15, color='#2980B9')

# Mark the optimal threshold (Youden's J)
j_scores = tpr - fpr
best_idx  = np.argmax(j_scores)
ax.scatter(fpr[best_idx], tpr[best_idx], s=120, color='red', zorder=5,
           label=f'Best Threshold = {thresholds[best_idx]:.2f}')
ax.annotate(f'  ({fpr[best_idx]:.2f}, {tpr[best_idx]:.2f})',
            (fpr[best_idx], tpr[best_idx]), fontsize=10, color='red')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.set_title('ROC Curve — Logistic Regression (Pass/Fail)')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.4)
plt.tight_layout()
savefig('07_roc_curve.png')

# ════════════════════════════════════════════════════════════
# PLOT 8 — Predicted vs Actual Score
# ════════════════════════════════════════════════════════════
print("[PLOT 8] Predicted vs Actual Score...")
r2   = r2_score(y_reg_test, y_pred_score)
mae  = mean_absolute_error(y_reg_test, y_pred_score)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_score))
residuals = y_reg_test - y_pred_score

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter: Predicted vs Actual
axes[0].scatter(y_reg_test, y_pred_score, color=BLUE, alpha=0.7,
                edgecolors='white', s=80)
mn = min(y_reg_test.min(), y_pred_score.min()) - 2
mx = max(y_reg_test.max(), y_pred_score.max()) + 2
axes[0].plot([mn,mx],[mn,mx], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'Predicted vs Actual Score\nR²={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}')
axes[0].legend()

# Residual Plot
axes[1].scatter(y_pred_score, residuals, color='#8E44AD', alpha=0.7,
                edgecolors='white', s=80)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].fill_between([y_pred_score.min()-2, y_pred_score.max()+2],
                     [-5,-5],[5,5], alpha=0.1, color='green',
                     label='±5 error band')
axes[1].set_xlabel('Predicted Score')
axes[1].set_ylabel('Residual (Actual − Predicted)')
axes[1].set_title('Residual Plot (Linear Regression)')
axes[1].legend()

plt.tight_layout()
savefig('08_predicted_vs_actual.png')

# ════════════════════════════════════════════════════════════
# PLOT 9 — Learning Curves
# ════════════════════════════════════════════════════════════
print("[PLOT 9] Learning Curves...")
X_all_sc = scaler.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, model, y_target, title, scorer in zip(
    axes,
    [LogisticRegression(max_iter=1000, random_state=42),
     LinearRegression()],
    [y_cls, y_reg],
    ['Logistic Regression (Classification)', 'Linear Regression (Score Prediction)'],
    ['accuracy', 'r2']
):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_all_sc, y_target,
        cv=5, scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42, shuffle=True
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color=BLUE, label='Training Score')
    ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std,
                    alpha=0.15, color=BLUE)
    ax.plot(train_sizes, val_mean, 's-', color=PASS_COLOR, label='Cross-Validation Score')
    ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std,
                    alpha=0.15, color=PASS_COLOR)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel(scorer.upper())
    ax.set_title(f'Learning Curve — {title}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
savefig('09_learning_curves.png')

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  ALL VISUALIZATIONS SAVED")
print("="*55)
all_plots = [
    '01_score_distribution.png',
    '02_study_hours_vs_score.png',
    '03_correlation_heatmap.png',
    '04_feature_importance.png',
    '05_boxplots_by_outcome.png',
    '06_confusion_matrix.png',
    '07_roc_curve.png',
    '08_predicted_vs_actual.png',
    '09_learning_curves.png',
]
for p in all_plots:
    path = os.path.join(REPORTS_DIR, p)
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING':6s} {p}")

print(f"\n  Classification Accuracy : {accuracy_score(y_cls_test, y_pred_cls)*100:.2f}%")
print(f"  ROC AUC                 : {roc_auc:.4f}")
print(f"  Regression R2 Score     : {r2_score(y_reg_test, y_pred_score):.4f}")
print(f"  MAE                     : {mean_absolute_error(y_reg_test, y_pred_score):.4f}")
print(f"  RMSE                    : {np.sqrt(mean_squared_error(y_reg_test, y_pred_score)):.4f}")
print("="*55)
