# prediction_model.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# -----------------------------
# Feature Engineering Function
# -----------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the engineered columns identified in the EDA.
    Aligns columns for prediction.
    """

    df = df.copy()

    # Binary / ordinal encoding
    df["payment_type"] = df["payment_type"].map(
        {"credit": 1.0, "debit": 2.0}
    ).astype(float)

    # Target encoding
    df["status"] = df["status"].map(
        {"Not_Canceled": 0, "Canceled": 1}
    ).astype(int)

    feature_columns = [
        "customer_collects",
        "is_a_repeat_order",
        "lead_time",
        "n_customer_notes",
        "n_items_above_quantity_10",
        "n_listed_addresses",
        "n_listed_payment_methods",
        "n_previous_cancelled_orders",
        "n_previous_completed_orders",
        "n_small_items",
        "payment_type",
        "total_price",
        "status"
    ]

    return df[feature_columns]


# -----------------------------
# Model Training & Prediction
# -----------------------------
def train_model(df: pd.DataFrame):
    """
    Train a Logistic Regression model and produce cancellation predictions.
    """

    df = prepare_features(df)

    X = df.drop("status", axis=1)
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression (interpretable & probability-based)
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # -----------------------------
    # Final Prediction Output (Text)
    # -----------------------------
    prediction_results = pd.DataFrame({
        "cancellation_risk_score": y_prob,
        "predicted_status": np.where(y_prob > 0.5, "Canceled", "Not_Canceled")
    })

    prediction_results_sorted = prediction_results.sort_values(
        by="cancellation_risk_score",
        ascending=False
    )

    print("\n--- Predicted Order Cancellations (Highest Risk First) ---")
    print(prediction_results_sorted.head(20).to_string(index=False))

    return model


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    data = pd.read_csv("orders_cleaned.csv")
    model = train_model(data)