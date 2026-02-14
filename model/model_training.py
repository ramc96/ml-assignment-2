"""
model_training.py
─────────────────
BITS Pilani | M.Tech AIML/DSE | Machine Learning – Assignment 2
Trains all 6 classification models on the Heart Disease dataset
and prints a full evaluation metrics table.

Usage:
    python model_training.py

Dataset:
    heart.csv  (Heart Disease Dataset from Kaggle)
    https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("BITS Pilani – ML Assignment 2")
print("Heart Disease Classification")
print("=" * 60)

df = pd.read_csv("heart.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"\nTarget distribution:\n{df['HeartDisease'].value_counts()}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. Define Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor":  KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(use_label_encoder=False,
                                         eval_metric='logloss',
                                         random_state=42, verbosity=0)
}

# ─────────────────────────────────────────────
# 4. Train & Evaluate
# ─────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n{'─'*50}")
    print(f"Training: {name}")

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        "Accuracy":  round(acc,  4),
        "AUC":       round(auc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4),
        "MCC":       round(mcc,  4)
    }

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

# ─────────────────────────────────────────────
# 5. Summary Table
# ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY – Evaluation Metrics for All Models")
print("=" * 80)
results_df = pd.DataFrame(results).T
print(results_df.to_string())
print("=" * 80)
print("\nBest Model by Accuracy :", results_df['Accuracy'].idxmax())
print("Best Model by F1 Score  :", results_df['F1 Score'].idxmax())
print("Best Model by AUC       :", results_df['AUC'].idxmax())
print("Best Model by MCC       :", results_df['MCC'].idxmax())
print("\nDone! ✅")
