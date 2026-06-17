"""
Project 2: Predictive Modeling Using Machine Learning
--------------------------------------------------------
Goal   : Build a model to predict outcomes (benign/malignant) based on
         medical measurement data.
Models : Logistic Regression (linear model), Decision Tree, Random Forest
Output : Accuracy/Precision/Recall/F1 table, confusion matrices, ROC curves.

Dataset: scikit-learn's built-in Breast Cancer Wisconsin dataset
         (real-world, no internet download needed). To use your own data,
         replace `load_data()` with `pd.read_csv("your_file.csv")` and set
         X / y accordingly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay)

np.random.seed(42)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")
    return X, y, data.target_names


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    rows = []
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        })
    return pd.DataFrame(rows), predictions


def plot_confusion_matrices(models, predictions, y_test, target_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved to: {path}")


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curves saved to: {path}")


def main():
    X, y, target_names = load_data()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    results, predictions = evaluate_models(models, X_test, y_test)

    print("\nModel Performance Comparison:")
    print(results.to_string(index=False))

    plot_confusion_matrices(models, predictions, y_test, target_names)
    plot_roc_curves(models, X_test, y_test)

    best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
    print(f"\nBest performing model on this test split: {best_model}")


if __name__ == "__main__":
    main()
