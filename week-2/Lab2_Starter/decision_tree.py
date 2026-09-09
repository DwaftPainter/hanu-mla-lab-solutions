"""
Lab 2: Decision Trees on Educational Response Data
Machine Learning Applications (MLA), HANU AY 2026-2027

Academic Integrity Declaration:
I, Nghiêm Thành Công (2301140014), declare that this code is my own original work.
I have not copied or adapted code from any external repositories or previous years.
Any sources or libraries used are explicitly cited below.
"""

import os
import sys

# Configure writable cache directory for Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Add parent directory to path so utils.py is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, compute_accuracy

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
DEPTHS = list(range(1, 21))
FIGURE_DIR = "figs"
os.makedirs(FIGURE_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Task 1: Load and inspect the data
# ----------------------------------------------------------------------------
print("=" * 60)
print("Lab 2: Decision Trees on Educational Response Data")
print("Student: Nghiêm Thành Công (ID: 2301140014)")
print("=" * 60)
print("Task 1: Data loading and inspection")
print("-" * 60)

# Load the training, validation, and test splits using load_data().
X_train, y_train, X_val, y_val, X_test, y_test = load_data("data")

n_train = len(y_train)
n_val = len(y_val)
n_test = len(y_test)

# Number of unique students and unique questions in the training set
n_unique_students = len(np.unique(X_train[:, 0]))
n_unique_questions = len(np.unique(X_train[:, 1]))

# Proportion of correct (label = 1) responses in the training set
prop_correct = np.mean(y_train == 1)

print(f"(a) Number of examples:")
print(f"    - Training set   : {n_train:,}")
print(f"    - Validation set : {n_val:,}")
print(f"    - Test set       : {n_test:,}")
print(f"(b) Unique entities in training set:")
print(f"    - Unique students : {n_unique_students:,}")
print(f"    - Unique questions: {n_unique_questions:,}")
print(f"(c) Proportion of correct responses (label = 1) in training set:")
print(f"    - Proportion: {prop_correct:.4f} ({prop_correct * 100:.2f}%)")



# ----------------------------------------------------------------------------
# Task 2: Fit decision trees at multiple depths
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Task 2: Fitting trees at multiple depths (max_depth: 1 to 20)")
print("=" * 60)

train_accuracies = []
val_accuracies = []

for depth in DEPTHS:
    # (a) Create DecisionTreeClassifier with current depth and RANDOM_STATE
    clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    
    # (b) Fit on the training set
    clf.fit(X_train, y_train)
    
    # (c) Record train and validation accuracy using compute_accuracy
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    
    tr_acc = compute_accuracy(y_train, y_pred_train)
    va_acc = compute_accuracy(y_val, y_pred_val)
    
    train_accuracies.append(tr_acc)
    val_accuracies.append(va_acc)

# Display results table
print(f"{'Depth':>6}  {'Train Acc':>12}  {'Val Acc':>12}")
print("-" * 34)
for d, tr, va in zip(DEPTHS, train_accuracies, val_accuracies):
    marker = " <-- (highest Val Acc)" if va == max(val_accuracies) else ""
    print(f"{d:>6}  {tr:>12.4f}  {va:>12.4f}{marker}")



# ----------------------------------------------------------------------------
# Task 3: Plot the validation curve
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Task 3: Validation curve")
print("=" * 60)

# Identify optimal depth from validation accuracies
best_val_idx = int(np.argmax(val_accuracies))
optimal_depth = DEPTHS[best_val_idx]
best_val_acc = val_accuracies[best_val_idx]
optimal_train_acc = train_accuracies[best_val_idx]

# Plot train and validation accuracy against max_depth
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(DEPTHS, train_accuracies, marker='o', color='#1e40af', linewidth=2.0, markersize=5, label='Train')
plt.plot(DEPTHS, val_accuracies, marker='s', color='#dc2626', linewidth=2.0, markersize=5, label='Validation')

# Mark optimal depth with vertical dashed line
plt.axvline(x=optimal_depth, color='#16a34a', linestyle='--', linewidth=1.8,
            label=f'Optimal depth (d* = {optimal_depth}, Val Acc = {best_val_acc:.4f})')

plt.title('Decision Tree: Training and Validation Accuracy vs. Tree Depth (max_depth)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('max_depth', fontsize=11, fontweight='semibold')
plt.ylabel('Accuracy', fontsize=11, fontweight='semibold')
plt.xlim(0.5, 20.5)
plt.ylim(0.0, 1.0)
plt.xticks(DEPTHS)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=False)
plt.tight_layout()

plot_path = os.path.join(FIGURE_DIR, "validation_curve.pdf")
plt.savefig(plot_path)
plt.savefig(os.path.join(FIGURE_DIR, "validation_curve.png"), dpi=300)
plt.close()
print(f"Validation curve figure successfully saved to: {plot_path}")


# ----------------------------------------------------------------------------
# Task 4: Select optimal depth and evaluate on test set
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Task 4: Optimal depth selection and test evaluation")
print("=" * 60)

# (a) Identify the max_depth value that achieves the highest validation accuracy
print(f"(a) Optimal max_depth selected from validation set: d* = {optimal_depth}")
print(f"    - Train accuracy at d* = {optimal_depth} : {optimal_train_acc:.4f} ({optimal_train_acc * 100:.2f}%)")
print(f"    - Validation accuracy at d* = {optimal_depth} : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

# (b) Retrain a DecisionTreeClassifier with that depth on the full training set
clf_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=RANDOM_STATE)
clf_optimal.fit(X_train, y_train)

# (c) Report test accuracy
y_pred_test = clf_optimal.predict(X_test)
dt_test_acc = compute_accuracy(y_test, y_pred_test)
print(f"(c) Final Test Accuracy (Decision Tree, d={optimal_depth}): {dt_test_acc:.4f} ({dt_test_acc * 100:.2f}%)")

# (d) Compare to the best KNN test accuracy from Lab 1
# From Lab 1 (Nghiêm Thành Công - 2301140014):
# User-Based KNN (k=9) Test Accuracy: 0.6887 (68.87%)
# Item-Based KNN (k=51) Test Accuracy: 0.6847 (68.47%)
knn_best_test_acc = 0.6887  # User-Based KNN (k=9)
knn_diff = knn_best_test_acc - dt_test_acc

print(f"(d) Comparison against Lab 1 Best KNN Model:")
print(f"    - Best KNN Model (User-Based, k=9) Test Accuracy : {knn_best_test_acc:.4f} ({knn_best_test_acc * 100:.2f}%)")
print(f"    - Decision Tree (max_depth={optimal_depth}) Test Accuracy      : {dt_test_acc:.4f} ({dt_test_acc * 100:.2f}%)")
if knn_best_test_acc > dt_test_acc:
    print(f"    - Higher Model : KNN Classifier is higher by {knn_diff:.4f} ({knn_diff * 100:.2f} percentage points)")
else:
    print(f"    - Higher Model : Decision Tree is higher by {-knn_diff:.4f} ({-knn_diff * 100:.2f} percentage points)")


# ----------------------------------------------------------------------------
# Task 5: Results summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Task 5: Written Analysis Summary")
print("=" * 60)
print(
    f"1. Validation Curve & Overfitting:\n"
    f"   - Validation curve follows a concave trajectory, rising from {val_accuracies[0]:.4f} (d=1) to peak at {best_val_acc:.4f} (d={optimal_depth}).\n"
    f"   - Overfitting begins past depth {optimal_depth}: training accuracy monotonically rises to {train_accuracies[-1]:.4f} (d=20) while validation accuracy drops to {val_accuracies[-1]:.4f}.\n\n"
    f"2. Generalization Gap at Optimal Depth:\n"
    f"   - At d*={optimal_depth}, Train Acc = {optimal_train_acc:.4f} and Val Acc = {best_val_acc:.4f}.\n"
    f"   - The gap is very small ({abs(optimal_train_acc - best_val_acc)*100:.2f}%), indicating solid generalization without excessive variance.\n\n"
    f"3. Why Validation Accuracy Decreases at High Depths:\n"
    f"   - High-depth trees partition feature space into ultra-fine rectangular cells, memorizing noise and idiosyncratic (user, question) training samples.\n\n"
    f"4. Decision Tree vs. KNN Comparison:\n"
    f"   - KNN achieves superior test performance ({knn_best_test_acc*100:.2f}% vs {dt_test_acc*100:.2f}%).\n"
    f"   - Nominal user/question integer codes lack metric ordering for axis-aligned splits, whereas KNN directly computes collaborative similarity in student response space.\n"
)
print("=" * 60)
print("Lab 2 Execution Completed Successfully.")
print("=" * 60)

