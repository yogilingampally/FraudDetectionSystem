"""
Credit card fraud detection: training script (v2).

What changed compared with the first version
--------------------------------------------
1. Three-way split (60% train / 20% validation / 20% test).
   Models and the threshold are chosen on VALIDATION data only.
   The test set is used once, at the end, for the final honest numbers.
2. Duplicate rows are dropped before splitting, so the same transaction
   cannot appear in both training and test data.
3. The exact feature column order is saved, so the app can never mix it up.
4. A small sample of the test set is saved (all fraud rows + some legit rows)
   so the Streamlit app can load real transactions without the 150 MB CSV.
5. Package versions, metrics and the winning model type are saved to a
   JSON file, so the app and requirements.txt match the training environment.

Run from the project folder:   python train_model.py
Needs:                         data/creditcard.csv
"""

import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ============================================================
# 0. SETTINGS (change these here, not inside the code below)
# ============================================================

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 20% of all data is the final test set
VALIDATION_SIZE = 0.25    # 25% of the remaining 80% = 20% of all data
DROP_DUPLICATES = True

MINIMUM_RECALL = 0.80     # selection rule: recall must be at least this
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

SAMPLE_LEGIT_ROWS = 2000  # legit rows saved for the app's sample mode

# The exact order the model is trained on, and the app must use.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def compute_metrics(y_true, y_prob, threshold):
    """Return a dictionary of metrics for one threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }


# ============================================================
# 1. LOAD DATASET AND CHECK IT
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDATA LOADED SUCCESSFULLY!")
print("Dataset shape:", df.shape)
print("*" * 60)

missing_columns = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN]
                   if c not in df.columns]
if missing_columns:
    raise ValueError(f"These expected columns are missing: {missing_columns}")

print("Missing values:", int(df.isnull().sum().sum()))
duplicate_count = int(df.duplicated().sum())
print("Duplicate rows:", duplicate_count)

if DROP_DUPLICATES and duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {duplicate_count} duplicate rows. New shape: {df.shape}")

print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts())
print("\nFraud percentage:")
print((df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(4))
print("*" * 60)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET (fixed column order)
# ============================================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ============================================================
# 3. SPLIT: TRAIN / VALIDATION / TEST (all stratified)
# ============================================================

# Step A: hold out the final test set. It is not touched until section 9.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Step B: split the rest into training and validation data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print("\nDATA SPLIT")
print("Training:  ", X_train.shape, "fraud cases:", int(y_train.sum()))
print("Validation:", X_val.shape, "fraud cases:", int(y_val.sum()))
print("Test:      ", X_test.shape, "fraud cases:", int(y_test.sum()))
print("*" * 60)


# ============================================================
# 4. CLASS IMBALANCE WEIGHT (from training data only)
# ============================================================

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("\nXGBoost scale_pos_weight:", round(float(scale_pos_weight), 2))
print("*" * 60)


# ============================================================
# 5. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ]),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=5,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# 6. TRAIN ON TRAINING DATA, COMPARE ON VALIDATION DATA
# ============================================================

comparison_results = []
validation_probabilities = {}

for name, model in models.items():

    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_val_prob = model.predict_proba(X_val)[:, 1]
    validation_probabilities[name] = y_val_prob

    results = {"Model": name, **compute_metrics(y_val, y_val_prob, 0.50)}
    comparison_results.append(results)

    print(f"{name} training completed!")
    print("Validation Precision:", round(results["Precision"], 4))
    print("Validation Recall:   ", round(results["Recall"], 4))
    print("Validation F1 Score: ", round(results["F1 Score"], 4))
    print("Validation PR-AUC:   ", round(results["PR-AUC"], 4))

comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values(by="PR-AUC", ascending=False)

print("\n\nMODEL COMPARISON ON VALIDATION DATA AT THRESHOLD 0.50")
print(comparison_df.to_string(index=False))
print("*" * 60)


# ============================================================
# 7. TRY DIFFERENT THRESHOLDS ON VALIDATION DATA
# ============================================================

threshold_results = []

for name, probabilities in validation_probabilities.items():
    for threshold in THRESHOLDS:
        metrics = compute_metrics(y_val, probabilities, threshold)
        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "False Positives": metrics["False Positives"],
            "False Negatives": metrics["False Negatives"],
            "True Positives": metrics["True Positives"],
            "True Negatives": metrics["True Negatives"],
        })

threshold_df = pd.DataFrame(threshold_results)

print("\n\nTHRESHOLD COMPARISON ON VALIDATION DATA (TOP 20 BY F1 SCORE)")
print(threshold_df.sort_values(by="F1 Score", ascending=False)
      .head(20).to_string(index=False))
print("*" * 60)


# ============================================================
# 8. SELECT THE BEST MODEL AND THRESHOLD (validation data only)
# ============================================================

# Rule: recall must be at least MINIMUM_RECALL, then maximize F1 and precision.
eligible = threshold_df[threshold_df["Recall"] >= MINIMUM_RECALL]

if not eligible.empty:
    best_configuration = eligible.sort_values(
        by=["F1 Score", "Precision"], ascending=False
    ).iloc[0]
    selection_note = f"recall >= {MINIMUM_RECALL} rule satisfied"
else:
    best_configuration = threshold_df.sort_values(
        by=["F1 Score", "Precision"], ascending=False
    ).iloc[0]
    selection_note = (f"no configuration reached recall >= {MINIMUM_RECALL}; "
                      "picked best F1 instead")

best_model_name = str(best_configuration["Model"])
best_threshold = float(best_configuration["Threshold"])
final_model = models[best_model_name]

print("\nSELECTED MODEL (CHOSEN ON VALIDATION DATA)")
print("Model:", best_model_name)
print("Threshold:", best_threshold)
print("Note:", selection_note)
print("Validation Precision:", round(best_configuration["Precision"], 4))
print("Validation Recall:   ", round(best_configuration["Recall"], 4))
print("Validation F1 Score: ", round(best_configuration["F1 Score"], 4))
print("*" * 60)


# ============================================================
# 9. FINAL, UNBIASED EVALUATION ON THE TEST SET (used once)
# ============================================================

y_test_prob = final_model.predict_proba(X_test)[:, 1]
test_metrics = compute_metrics(y_test, y_test_prob, best_threshold)
y_test_pred = (y_test_prob >= best_threshold).astype(int)

print("\nFINAL RESULTS ON TEST DATA (NOT USED FOR ANY CHOICE ABOVE)")
print("Selected Model:", best_model_name)
print("Selected Threshold:", best_threshold)
for key in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]:
    print(f"{key}: {test_metrics[key]:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))
print("*" * 60)


# ============================================================
# 10. SAVE EVERYTHING THE STREAMLIT APP NEEDS
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 10a. Model and threshold
joblib.dump(final_model, MODEL_DIR / "fraud_model.pkl")
joblib.dump(best_threshold, MODEL_DIR / "fraud_threshold.pkl")

# 10b. Exact feature order (the app must build its input in this order)
joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.pkl")

# 10c. Comparison tables (validation data)
comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
threshold_df.to_csv(MODEL_DIR / "threshold_comparison.csv", index=False)

# 10d. Sample transactions for the app: all test fraud rows + some legit rows
test_fraud = X_test[y_test == 1].assign(Class=1)
legit_pool = X_test[y_test == 0]
test_legit = legit_pool.sample(
    n=min(SAMPLE_LEGIT_ROWS, len(legit_pool)),
    random_state=RANDOM_STATE,
).assign(Class=0)

sample_transactions = pd.concat([test_fraud, test_legit])
sample_transactions = sample_transactions.sample(
    frac=1, random_state=RANDOM_STATE
)
sample_transactions.to_csv(MODEL_DIR / "sample_transactions.csv", index=False)

# 10e. Metadata: model type, metrics, split sizes and package versions
tree_models = {"Decision Tree", "Random Forest", "XGBoost"}

metadata = {
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "model_name": best_model_name,
    "threshold": best_threshold,
    "selection_note": selection_note,
    "explainer_hint": "tree" if best_model_name in tree_models else "linear",
    "is_pipeline": isinstance(final_model, Pipeline),
    "feature_columns": FEATURE_COLUMNS,
    "split": {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_fraud": int(y_train.sum()),
        "validation_fraud": int(y_val.sum()),
        "test_fraud": int(y_test.sum()),
        "duplicates_dropped": duplicate_count if DROP_DUPLICATES else 0,
    },
    "validation_metrics": {
        k: (float(v) if not isinstance(v, int) else v)
        for k, v in compute_metrics(
            y_val, validation_probabilities[best_model_name], best_threshold
        ).items()
    },
    "test_metrics": {
        k: (float(v) if not isinstance(v, int) else v)
        for k, v in test_metrics.items()
    },
    "versions": {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}

with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 10f. Pinned versions for requirements.txt (add streamlit when building the app)
with open(MODEL_DIR / "requirements_training.txt", "w") as f:
    f.write(f"scikit-learn=={sklearn.__version__}\n")
    f.write(f"xgboost=={xgboost.__version__}\n")
    f.write(f"pandas=={pd.__version__}\n")
    f.write(f"numpy=={np.__version__}\n")
    f.write(f"joblib=={joblib.__version__}\n")

print("\nALL FILES SAVED SUCCESSFULLY IN:", MODEL_DIR.resolve())
for file_name in [
    "fraud_model.pkl", "fraud_threshold.pkl", "feature_columns.pkl",
    "model_comparison.csv", "threshold_comparison.csv",
    "sample_transactions.csv", "model_metadata.json",
    "requirements_training.txt",
]:
    print(" -", file_name)

print("\nMODEL TRAINING AND COMPARISON COMPLETED!")