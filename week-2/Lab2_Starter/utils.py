"""
utils.py: data loading and evaluation utilities for MLA Labs 1 and 2.
Machine Learning Applications (MLA), HANU AY 2026-2027

This module provides functions to load the Eedi student response dataset and
compute standard evaluation metrics. It is shared across Lab 1 and Lab 2 and
should not be modified by students.

Note on filenames: the functions below expect the standard Eedi split filenames
used in the course data directory. If your local copies use different names
(e.g., 'train.csv' instead of 'train_data.csv'), update the filename constants
at the top of each function accordingly.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir: str = "data"):
    """
    Load the Eedi student response dataset from CSV files.

    Expects three files in data_dir:
        train_data.csv  : training split
        valid_data.csv  : validation split
        test_data.csv   : test split

    Each file must contain at minimum the columns:
        user_id       : integer or string student identifier
        question_id   : integer or string question identifier
        is_correct    : binary label (1 = correct, 0 = incorrect)

    If the label column is named 'correct' rather than 'is_correct', the
    function falls back to 'correct' automatically.

    Features returned are [user_id_code, question_id_code], where each ID has
    been integer-encoded using a LabelEncoder fitted on the training set.
    Unseen IDs in validation and test splits are mapped to -1.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : numpy arrays
        X arrays have shape (n_samples, 2); y arrays have shape (n_samples,).
    """
    train_path = os.path.join(data_dir, "train_data.csv")
    valid_path = os.path.join(data_dir, "valid_data.csv")
    test_path  = os.path.join(data_dir, "test_data.csv")

    for path in (train_path, valid_path, test_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected data file not found: {path}\n"
                "Ensure the 'data/' directory contains train_data.csv, "
                "valid_data.csv, and test_data.csv."
            )

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df  = pd.read_csv(test_path)

    # Resolve label column name
    label_col = "is_correct" if "is_correct" in train_df.columns else "correct"
    for df, name in ((train_df, "train"), (valid_df, "valid"), (test_df, "test")):
        if label_col not in df.columns:
            raise KeyError(
                f"Neither 'is_correct' nor 'correct' found in {name} split columns: "
                f"{list(df.columns)}"
            )

    # Encode user_id and question_id as integer codes
    user_enc = LabelEncoder().fit(train_df["user_id"])
    question_enc = LabelEncoder().fit(train_df["question_id"])

    def encode_split(df, user_enc, question_enc):
        user_ids = df["user_id"].values
        question_ids = df["question_id"].values

        # Map unseen IDs to -1
        user_mask = np.isin(user_ids, user_enc.classes_)
        question_mask = np.isin(question_ids, question_enc.classes_)

        user_codes = np.where(
            user_mask,
            user_enc.transform(np.where(user_mask, user_ids, user_enc.classes_[0])),
            -1
        )
        question_codes = np.where(
            question_mask,
            question_enc.transform(
                np.where(question_mask, question_ids, question_enc.classes_[0])
            ),
            -1
        )
        X = np.column_stack([user_codes, question_codes])
        y = df[label_col].values.astype(int)
        return X, y

    X_train, y_train = encode_split(train_df, user_enc, question_enc)
    X_val,   y_val   = encode_split(valid_df, user_enc, question_enc)
    X_test,  y_test  = encode_split(test_df,  user_enc, question_enc)

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Fraction of predictions that match the ground-truth labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}."
        )
    return float(np.mean(y_true == y_pred))


def load_student_metadata(data_dir: str = "data") -> pd.DataFrame:
    """
    Load student metadata from student_meta.csv if the file is present.

    The Eedi student_meta.csv typically contains columns such as:
        user_id, gender, premium_pupil, and age group indicators.

    Parameters
    ----------
    data_dir : str
        Path to the directory that may contain student_meta.csv.

    Returns
    -------
    pd.DataFrame
        The metadata table, or an empty DataFrame if the file is not found.
        A warning is printed when the file is absent so callers are informed.
    """
    meta_path = os.path.join(data_dir, "student_meta.csv")
    if not os.path.isfile(meta_path):
        print(
            f"Warning: student_meta.csv not found in '{data_dir}'. "
            "Returning an empty DataFrame."
        )
        return pd.DataFrame()

    return pd.read_csv(meta_path)
