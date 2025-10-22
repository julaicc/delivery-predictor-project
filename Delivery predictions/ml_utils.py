# ml_utils.py
# Utilities for preprocessing, feature engineering, encoding, training, and single-row prediction
# Drops Gender/ID, dropna
# Renames: Reached.on.Time_Y.N: Reached_on_time, Cost_of_the_Product: Product_Cost, Mode_of_Shipment: Shipment_mode
# Engineers: Estimated_Delivery_Time, Simulated_Distance_km
# One-hot encodes: Warehouse_block, Shipment_mode, Product_importance
# Builds X, y

from __future__ import annotations
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ---------- Shipment Mode constants ala delivery_predict.ipynb ----------
MODE_TIME: Dict[str, float] = {"Ship": 12.0, "Road": 6.0, "Flight": 2.0}
MODE_MULTIPLIER: Dict[str, float] = {"Ship": 1.8, "Road": 1.0, "Flight": 0.6}

# Block Distance adding E= 850 so FE still works if E appears.
BLOCK_DISTANCE: Dict[str, float] = {"A": 150.0, "B": 300.0, "C": 500.0, "D": 700.0, "E": 850.0, "F": 1000.0}

# Categoricals Variables encoding
CATEGORICAL_COLS: List[str] = ["Warehouse_block", "Shipment_mode", "Product_importance"]


# Renaming headers                                                                ***Ver si es necesario escribir todos los encabezados
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to the exact headers your code expects."""
    rename_map = {
        "Reached.on.Time_Y.N": "Reached_on_time",
        "Cost_of_the_Product": "Product_Cost",
        "Mode_of_Shipment": "Shipment_mode",
        "Cost_of_the product": "Product_Cost",
        "Customer_care_calls": "Customer_care_calls",
        "Customer_rating": "Customer_rating",
        "Prior_purchases": "Prior_purchases",
        "Discount_offered": "Discount_offered",
        "Weight_in_gms": "Weight_in_gms",
        "Warehouse_block": "Warehouse_block",
        "Product_importance": "Product_importance",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})



#  Preprocessing & Feature Engineering

def preprocess_like_notebook(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce your notebook steps:
      - drop Gender, ID
      - dropna
      - rename key columns
      - engineer Estimated_Delivery_Time & Simulated_Distance_km
      - add DeliveryStatus label for EDA
    """

    df = df_raw.copy()

    # 1) Drop irrelevant columns
    for col in ("Gender", "ID"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2) Column renaming
    df.rename(
        columns={
            "Reached.on.Time_Y.N": "Reached_on_time",
            "Cost_of_the_Product": "Product_Cost",
            "Mode_of_Shipment": "Shipment_mode",
        },
        inplace=True,
    )


    # 3) Drop missing rows (simple approach used in your code)
    df.dropna(inplace=True)

    # Sanity checks for required columns
    required = [
        "Weight_in_gms",
        "Discount_offered",
        "Prior_purchases",
        "Customer_care_calls",
        "Shipment_mode",
        "Warehouse_block",
        "Reached_on_time",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for preprocessing: {missing}")


    # 4) Estimated_Delivery_Time
    df["Estimated_Delivery_Time"] = (
        df["Weight_in_gms"] / 1000.0 * 0.5
        + df["Discount_offered"] * 0.2
        + df["Prior_purchases"] * -0.3
        + df["Customer_care_calls"] * 1.2
    )
    df["Estimated_Delivery_Time"] += df["Shipment_mode"].map(MODE_TIME)


    # 5) Simulated_Distance_km
    df["BaseDistance"] = df["Warehouse_block"].map(BLOCK_DISTANCE)
    df["Simulated_Distance_km"] = df["BaseDistance"] * df["Shipment_mode"].map(MODE_MULTIPLIER)

    # Optional: drop helper column if you don't need it later
    df.drop(columns=["BaseDistance"], inplace=True)


    # 6) Human-friendly label for EDA
    df["DeliveryStatus"] = df["Reached_on_time"].map({0: "On Time", 1: "Late"})

    return df


    # Encoding & dataset split
def encode_and_build_xy(df_fe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    One-hot encodes the categorical columns (drop_first=True),
    then builds X (features) and y (target) exactly like your notebook.
    """
    df = df_fe.copy()
    if "Reached_on_time" not in df.columns:
        raise ValueError("Target column 'Reached_on_time' is missing.")

    # One-hot encode categoricals used in your code
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Exact feature list from your notebook
    base_features = [
        "Customer_care_calls",
        "Customer_rating",
        "Product_Cost",
        "Prior_purchases",
        "Discount_offered",
        "Weight_in_gms",
        "Estimated_Delivery_Time",
        "Simulated_Distance_km",
    ]
    dummy_prefixes = ["Warehouse_block_", "Shipment_mode_", "Product_importance_"]
    dummy_features = [c for c in df.columns if any(p in c for p in dummy_prefixes)]

    features = [c for c in (base_features + dummy_features) if c in df.columns]

    X = df[features]
    y = df["Reached_on_time"]

    return X, y, features


# Training & evaluation 
def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Tuple[RandomForestClassifier, dict, float, np.ndarray, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Train a RandomForest exactly like your notebook and return:
      - model
      - classification_report (dict)
      - ROC AUC
      - confusion matrix
      - (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return rf, report, auc, cm, (X_train, X_test, y_train, y_test)

    

  # Single-row "what-if" prediction 
def predict_one(
    rf: RandomForestClassifier,
    feature_cols: List[str],
    *,
    Warehouse_block: str,
    Shipment_mode: str,
    Product_importance: str,
    Customer_care_calls: int,
    Customer_rating: int,
    Product_Cost: float,
    Prior_purchases: int,
    Discount_offered: float,
    Weight_in_gms: float,
) -> Tuple[float, int]:
    """
    Build a single-row frame from raw inputs, apply the SAME feature engineering & encoding,
    align to training feature columns, and return (prob_late, pred_class).
    """
    tmp = pd.DataFrame(
        [
            {
                "Warehouse_block": Warehouse_block,
                "Shipment_mode": Shipment_mode,
                "Product_importance": Product_importance,
                "Customer_care_calls": Customer_care_calls,
                "Customer_rating": Customer_rating,
                "Product_Cost": Product_Cost,
                "Prior_purchases": Prior_purchases,
                "Discount_offered": Discount_offered,
                "Weight_in_gms": Weight_in_gms,
                # dummy placeholder to satisfy downstream code if needed
                "Reached_on_time": 0,
            }
        ]
    )
    # Reuse the same FE logic (inline to avoid needing the target)
    tmp["Estimated_Delivery_Time"] = (
        tmp["Weight_in_gms"] / 1000.0 * 0.5
        + tmp["Discount_offered"] * 0.2
        + tmp["Prior_purchases"] * -0.3
        + tmp["Customer_care_calls"] * 1.2
    )
    tmp["Estimated_Delivery_Time"] += tmp["Shipment_mode"].map(MODE_TIME)

    tmp["BaseDistance"] = tmp["Warehouse_block"].map(BLOCK_DISTANCE)
    tmp["Simulated_Distance_km"] = tmp["BaseDistance"] * tmp["Shipment_mode"].map(MODE_MULTIPLIER)
    tmp.drop(columns=["BaseDistance"], inplace=True)

    # One-hot encode like training (drop_first=True)
    tmp = pd.get_dummies(tmp, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to training feature columns (add missing as 0, keep order)
    aligned = pd.DataFrame(columns=feature_cols)
    for c in feature_cols:
        aligned[c] = tmp[c] if c in tmp.columns else 0
    aligned = aligned[feature_cols].astype(float)

    prob_late = rf.predict_proba(aligned)[:, 1][0]
    pred_class = int(rf.predict(aligned)[0])
    return prob_late, pred_class