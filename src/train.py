"""Compares a few models for predicting equipment faults.

Faults are rare, so plain accuracy is misleading. I look at precision (how
many alarms were real), recall (how many faults we caught) and F1.
"""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/readings.csv")
    X = df.drop(columns="fault")
    y = df["fault"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   min_samples_leaf=5, random_state=42)),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=42)),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.joblib")
        preds = model.predict(X_test)
        fitted[name] = preds
        rows.append({
            "model": name,
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds), 3),
            "f1": round(f1_score(y_test, preds), 3),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("results/scores.csv", index=False)

    # confusion matrix for the model with the best F1
    best = table.sort_values("f1", ascending=False).iloc[0]["model"]
    print("\nbest model by F1:", best)
    ConfusionMatrixDisplay.from_predictions(y_test, fitted[best])
    plt.title(best)
    plt.savefig("results/confusion_matrix.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
