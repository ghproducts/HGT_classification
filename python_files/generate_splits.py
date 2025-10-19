import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


positive_file = "2mers/mito_positive.csv"
negative_file = "2mers/mito_negative.csv"
random_seed =  42

def generate_data():
    mito_positive = pd.read_csv(positive_file)
    mito_positive["mito_positive"] = 1

    mito_negative = pd.read_csv(negative_file)
    mito_negative["mito_positive"] = 0


    combined = pd.concat([mito_positive, mito_negative])

    # select features and labels
    X = combined.drop(columns=["mito_positive","ID"])
    y = combined["mito_positive"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2, 
            stratify = y,
            random_state=random_seed
            )
    X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2, 
            stratify = y_train,
            random_state=random_seed
            )

    return X_train, X_test, X_val, y_train, y_test, y_val

def train_xgb(X_train, X_val, y_train, y_val):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg/pos) if pos > 0 else 1

    clf = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_seed,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50
            )

    clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
            )

    return clf

def evaluate_model(clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=3)

    print('====== test metrics =====')
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


def show_top_features(clf, top_k=20):
    # Works with sklearn API
    if hasattr(clf, "feature_importances_") and hasattr(clf, "feature_names_in_"):
        importances = clf.feature_importances_
        names = clf.feature_names_in_
        order = np.argsort(importances)[::-1][:top_k]
        print(f"\n=== Top {top_k} features (gain-based importance) ===")
        for i in order:
            print(f"{names[i]:>8s}: {importances[i]:.5f}")
    else:
        print("Feature importances not available.")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = generate_data()

    clf = train_xgb(X_train, X_val, y_train, y_val)

    evaluate_model(clf, X_test, y_test)
    show_top_features(clf, top_k=30)

    # Save the model (JSON is portable; you can also use joblib/pickle)
    #model_path = "xgb_mito.json"
    #clf.get_booster().save_model(model_path)
    #print(f"\nSaved model to {model_path}")



if __name__ == "__main__":
	main()
