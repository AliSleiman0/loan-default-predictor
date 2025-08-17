# src/train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from src.preprocess import clean_raw_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score



def train_and_save(
    raw_csv_path: str = "data/raw/loan_train.csv",
    out_model_path: str = "models/model.pkl",
    out_features_path: str = "models/features.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100
):
    # 1) Load raw CSV
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path)

    # 2) Clean using existing preprocess function (this returns a DataFrame expected by your pipeline)
    df = clean_raw_df(df_raw)

    # 3) Prepare target y and features X
    if 'Loan_Status' not in df.columns:
        raise ValueError("'Loan_Status' not found in cleaned dataframe. Ensure target column exists in raw CSV.")
    # If Loan_Status still has Y/N map it; if already numeric (0/1) this will keep it
    if df['Loan_Status'].dtype == object or df['Loan_Status'].dropna().isin(['Y','N']).any():
        y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    else:
        y = df['Loan_Status'].astype(int)

    # Drop ID and target; errors='ignore' in case they are missing
    X = df.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')

    # 4) Convert any remaining object/category columns to one-hot (these are columns that preprocess didn't encode)
    # This makes the training consistent: model only receives numeric columns.
    # Note: keep drop_first=False to preserve all categories (we will use this same scheme at inference)
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)

    # 5) Ensure no remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise RuntimeError(f"Non-numeric columns still present after get_dummies: {non_numeric}")

    # 6) Align column order (sort for reproducibility) OR keep natural order. We'll keep natural order but save it.
    feature_cols = X.columns.tolist()

    # 7) Train/test split (stratify to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 8) Train RandomForestClassifier on cleaned, numeric X
    pipeline = Pipeline(steps=[
    ("classifier", LogisticRegression(max_iter=1000))
])
# Param grid
    param_grid = {
       "classifier__C": [0.1, 1, 10],
       "classifier__penalty": ["l1", "l2"],
       "classifier__solver": ["liblinear"]  # compatible with l1 penalty
    }

# Grid search
    clf = GridSearchCV(
       pipeline,
       param_grid,
       cv=5,
       scoring="roc_auc",
       n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 9) Evaluate: use predicted probabilities for AUC
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"[train_model] Validation AUC: {auc:.4f}")

    # Optional: print classification report with default 0.5 threshold
    preds = clf.predict(X_test)
    print("[train_model] Classification report (test):")
    print(classification_report(y_test, preds))

    # 10) Save model and feature list
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    joblib.dump(clf, out_model_path)
    joblib.dump(feature_cols, out_features_path)
    print(f"[train_model] Saved model to: {out_model_path}")
    print(f"[train_model] Saved feature list to: {out_features_path}")

    return clf, feature_cols, auc


if __name__ == "__main__":
    # Adjust paths as needed
    raw_csv = "data/raw/loan_train.csv"
    model_path = "models/model.pkl"
    features_path = "models/features.pkl"
    train_and_save(raw_csv, model_path, features_path)
