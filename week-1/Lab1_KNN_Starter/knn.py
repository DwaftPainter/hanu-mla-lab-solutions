# MLA Academic Year 2026-2027 - Hanoi University (FIT-HANU)
# Lab 1: K-Nearest Neighbours on Educational Response Data
# Academic Integrity Declaration:
# I, Nghiêm Thành Công (2301140014), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
    evaluate,
)


def user_knn_predict_hanu(matrix, valid_data, k, return_confusion=False):
    """
    Predict missing values using user-based k-nearest neighbors (KNN).
    
    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing entries.
        valid_data: dict with 'user_id', 'question_id', 'is_correct'.
        k: int, number of nearest neighbors.
        return_confusion: bool, if True also return sklearn confusion matrix.
        
    Returns:
        accuracy: float, accuracy score on valid_data.
        (optional) conf_matrix: 2x2 ndarray confusion matrix.
    """
    # Initialize KNN imputer with k neighbors
    nbrs = KNNImputer(n_neighbors=k)
    # Fit and transform the user x question matrix
    mat = nbrs.fit_transform(matrix)
    
    # Get predictions for validation data
    pred = []
    actual = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        pred.append(1 if mat[u, q] >= 0.5 else 0)
        actual.append(valid_data["is_correct"][i])
    
    accuracy = np.mean(np.array(pred) == np.array(actual))
    
    if return_confusion:
        conf_matrix = confusion_matrix(actual, pred)
        return accuracy, conf_matrix
    
    return accuracy


def item_knn_predict_hanu(matrix, valid_data, k, student_id="2301140014"):
    """
    Predict missing values using item-based k-nearest neighbors (KNN).
    Also saves validation predictions to file named '{student_id}_item_knn_preds.npy'.
    
    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing entries.
        valid_data: dict with 'user_id', 'question_id', 'is_correct'.
        k: int, number of nearest neighbors.
        student_id: string, student ID for saving prediction array.
        
    Returns:
        accuracy: float, accuracy score on valid_data.
    """
    # Transpose matrix for item-based approach (questions x users)
    matrix_t = matrix.T
    
    # Initialize KNN imputer
    nbrs = KNNImputer(n_neighbors=k)
    # Fit and transform the transposed matrix
    mat_t = nbrs.fit_transform(matrix_t)
    
    # Transpose back to original shape (users x questions)
    mat = mat_t.T
    
    # Get predictions for validation data
    pred = []
    actual = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        pred.append(1 if mat[u, q] >= 0.5 else 0)
        actual.append(valid_data["is_correct"][i])
    
    # Save predictions if student_id is provided
    if student_id:
        pred_array = np.array(pred)
        np.save(f"{student_id}_item_knn_preds.npy", pred_array)
    
    accuracy = np.mean(np.array(pred) == np.array(actual))
    return accuracy


def main():
    print("==================================================")
    print(" FIT-HANU Machine Learning Applications (MLA)")
    print(" Lab 1: KNN on Educational Response Data")
    print(" Student: Nghiêm Thành Công (ID: 2301140014)")
    print("==================================================")

    # ---------------------------------------------------------
    # Task 1 - Dataset Inspection
    # ---------------------------------------------------------
    print("\n--- Task 1: Dataset Inspection ---")
    train_data = load_train_csv("./data")
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    sparse_matrix = load_train_sparse("./data").toarray()

    n_train_triples = len(train_data["user_id"])
    n_valid_triples = len(valid_data["user_id"])
    n_test_triples = len(test_data["user_id"])
    n_users, n_questions = sparse_matrix.shape
    total_entries = n_users * n_questions
    nan_entries = np.count_nonzero(np.isnan(sparse_matrix))
    sparsity_fraction = nan_entries / total_entries

    print(f"1. Number of Training Triples: {n_train_triples:,}")
    print(f"   Number of Validation Triples: {n_valid_triples:,}")
    print(f"   Number of Public Test Triples: {n_test_triples:,}")
    print(f"2. Sparse Matrix Shape: ({n_users} students, {n_questions} questions)")
    print(f"3. Sparsity Fraction: {sparsity_fraction:.4f} ({sparsity_fraction*100:.2f}%)")

    # Majority vote baseline calculation
    correct_map = {}
    total_map = {}
    for i, q in enumerate(train_data["question_id"]):
        correct_map[q] = correct_map.get(q, 0) + (1 if train_data["is_correct"][i] == 1 else 0)
        total_map[q] = total_map.get(q, 0) + 1

    mv_val_preds = [1.0 if (correct_map.get(q, 0) / float(total_map.get(q, 1))) >= 0.5 else 0.0 for q in valid_data["question_id"]]
    mv_test_preds = [1.0 if (correct_map.get(q, 0) / float(total_map.get(q, 1))) >= 0.5 else 0.0 for q in test_data["question_id"]]
    mv_val_acc = evaluate(valid_data, mv_val_preds)
    mv_test_acc = evaluate(test_data, mv_test_preds)
    print(f"4. Majority-Vote Baseline: Val Acc = {mv_val_acc:.4f} ({mv_val_acc*100:.2f}%), Test Acc = {mv_test_acc:.4f} ({mv_test_acc*100:.2f}%)")

    # ---------------------------------------------------------
    # Task 2 - User-Based KNN
    # ---------------------------------------------------------
    print("\n--- Task 2: User-Based KNN Experiments ---")
    k_values = [1, 3, 5, 7, 9, 11, 21, 51]
    
    user_val_accs = []
    user_train_accs = []
    user_imputed_matrices = {}
    best_k_user = None
    best_val_acc_user = -1.0

    for k in k_values:
        nbrs = KNNImputer(n_neighbors=k)
        mat = nbrs.fit_transform(sparse_matrix)
        val_acc = sparse_matrix_evaluate(valid_data, mat)
        train_acc = sparse_matrix_evaluate(train_data, mat)
        
        user_val_accs.append(val_acc)
        user_train_accs.append(train_acc)
        user_imputed_matrices[k] = mat
        
        print(f"k = {k:2d} | Train Acc: {train_acc:.4f} | Validation Acc: {val_acc:.5f}")
        if val_acc > best_val_acc_user:
            best_val_acc_user = val_acc
            best_k_user = k

    print(f"Optimal k for User-Based KNN: k* = {best_k_user} (Val Acc: {best_val_acc_user:.5f})")

    # ROC-AUC and Confusion Matrix for Best User KNN
    best_user_mat = user_imputed_matrices[best_k_user]
    pred_probs_user = [best_user_mat[valid_data["user_id"][i], valid_data["question_id"][i]] for i in range(len(valid_data["is_correct"]))]
    user_auc = roc_auc_score(valid_data["is_correct"], pred_probs_user)
    _, user_conf = user_knn_predict_hanu(sparse_matrix, valid_data, best_k_user, return_confusion=True)
    print(f"Best User KNN ROC-AUC: {user_auc:.4f}")
    print("Best User KNN Confusion Matrix:")
    print(user_conf)

    # ---------------------------------------------------------
    # Task 3 - Item-Based KNN
    # ---------------------------------------------------------
    print("\n--- Task 3: Item-Based KNN Experiments ---")
    matrix_t = sparse_matrix.T
    item_val_accs = []
    item_train_accs = []
    item_imputed_matrices = {}
    best_k_item = None
    best_val_acc_item = -1.0

    for k in k_values:
        nbrs = KNNImputer(n_neighbors=k)
        mat_t = nbrs.fit_transform(matrix_t)
        mat = mat_t.T
        
        val_acc = sparse_matrix_evaluate(valid_data, mat)
        train_acc = sparse_matrix_evaluate(train_data, mat)
        
        item_val_accs.append(val_acc)
        item_train_accs.append(train_acc)
        item_imputed_matrices[k] = mat
        
        print(f"k = {k:2d} | Train Acc: {train_acc:.4f} | Validation Acc: {val_acc:.5f}")
        if val_acc > best_val_acc_item:
            best_val_acc_item = val_acc
            best_k_item = k

    # Save item predictions
    item_knn_predict_hanu(sparse_matrix, valid_data, best_k_item, student_id="2301140014")
    print(f"Optimal k for Item-Based KNN: k* = {best_k_item} (Val Acc: {best_val_acc_item:.5f})")

    # ---------------------------------------------------------
    # Task 4 - Final Test Accuracy Evaluation
    # ---------------------------------------------------------
    print("\n--- Task 4: Final Test Accuracy ---")
    test_acc_user = sparse_matrix_evaluate(test_data, user_imputed_matrices[best_k_user])
    test_acc_item = sparse_matrix_evaluate(test_data, item_imputed_matrices[best_k_item])

    print(f"Majority-Vote Baseline Test Acc : {mv_test_acc:.4f} ({mv_test_acc*100:.2f}%)")
    print(f"User-Based KNN (k={best_k_user}) Test Acc : {test_acc_user:.4f} ({test_acc_user*100:.2f}%)")
    print(f"Item-Based KNN (k={best_k_item}) Test Acc : {test_acc_item:.4f} ({test_acc_item*100:.2f}%)")

    # ---------------------------------------------------------
    # Save Figures for Report
    # ---------------------------------------------------------
    os.makedirs("./figures", exist_ok=True)
    
    # User KNN Plot
    plt.figure(figsize=(6.2, 3.6), dpi=300)
    plt.plot(k_values, [a*100 for a in user_val_accs], marker='o', color='#1e3a8a', linewidth=2, label='User-Based Validation Acc (%)')
    plt.axvline(x=best_k_user, color='red', linestyle='--', alpha=0.8, label=f'Best k={best_k_user} ({best_val_acc_user*100:.2f}%)')
    plt.title('User-Based KNN: Validation Accuracy vs. k')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy (%)')
    plt.xticks(k_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/user_knn_val_curve.png')
    plt.close()

    # Item KNN Plot
    plt.figure(figsize=(6.2, 3.6), dpi=300)
    plt.plot(k_values, [a*100 for a in item_val_accs], marker='s', color='#15803d', linewidth=2, label='Item-Based Validation Acc (%)')
    plt.axvline(x=best_k_item, color='red', linestyle='--', alpha=0.8, label=f'Best k={best_k_item} ({best_val_acc_item*100:.2f}%)')
    plt.title('Item-Based KNN: Validation Accuracy vs. k')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy (%)')
    plt.xticks(k_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/item_knn_val_curve.png')
    plt.close()

    print("\n[Done] Figures and predictions saved successfully.")


if __name__ == "__main__":
    main()
