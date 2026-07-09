"""
DecodeLabs - Week 2 - Project 2: Data Classification Using AI
Fully self-contained (uses sklearn's built-in Iris dataset).
Run: python Week2.py
Requires: pip install scikit-learn matplotlib numpy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, ConfusionMatrixDisplay,
)


def load_and_understand_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names, target_names = iris.feature_names, iris.target_names

    print("Dataset shape:", X.shape)
    print("Features:", feature_names)
    print("Classes:", list(target_names))
    print("Samples per class:", np.bincount(y))
    print()
    return X, y, feature_names, target_names


def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test, scaler


def find_best_k(X_train, y_train, X_test, y_test, max_k=20):
    errors = []
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors.append(np.mean(preds != y_test))
    best_k = int(np.argmin(errors)) + 1
    print(f"Best K found for KNN: {best_k} (error rate: {errors[best_k - 1]:.2%})")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_k + 1), errors, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
    plt.xlabel("K value"); plt.ylabel("Error rate")
    plt.title("Tuning the Engine: Choosing K"); plt.legend()
    plt.tight_layout(); plt.savefig("k_tuning_curve.png", dpi=150); plt.close()
    return best_k


def compare_algorithms(X_train, y_train, X_test, y_test, best_k):
    models = {
        f"KNN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    results, trained_models = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"accuracy": acc, "f1": f1, "predictions": preds}
        trained_models[name] = model
        print(f"{name}: accuracy={acc:.2%}, F1={f1:.2%}")

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylim(0, 1.05); plt.ylabel("Accuracy")
    plt.title("Algorithm Comparison on Iris Test Set"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig("algorithm_comparison.png", dpi=150); plt.close()

    return results, trained_models


def evaluate_best_model(y_test, predictions, target_names, model_name):
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    cm = confusion_matrix(y_test, predictions)

    print(f"\nBest model: {model_name}")
    print(f"Accuracy: {acc:.2%} | F1 Score: {f1:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues"); plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()


def test_on_new_samples(model, scaler, target_names):
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.9, 3.1, 5.4, 2.1],
    ])
    predictions = model.predict(scaler.transform(new_samples))
    print("\nTesting on brand-new, unseen data:")
    for sample, pred in zip(new_samples, predictions):
        print(f"Measurements {sample} -> Predicted: {target_names[pred]}")


def main():
    X, y, feature_names, target_names = load_and_understand_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)

    best_k = find_best_k(X_train, y_train, X_test, y_test)
    results, trained_models = compare_algorithms(X_train, y_train, X_test, y_test, best_k)

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_predictions = results[best_name]["predictions"]
    best_model = trained_models[best_name]

    evaluate_best_model(y_test, best_predictions, target_names, best_name)
    test_on_new_samples(best_model, scaler, target_names)

    print("\nSaved charts: k_tuning_curve.png, algorithm_comparison.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
