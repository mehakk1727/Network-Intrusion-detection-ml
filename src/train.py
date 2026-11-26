"""Train and compare baseline models for binary network-threat detection."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_utils import binary_target, clean_flow_data, load_flow_data


def train(data_dir: str, output_dir: str, max_rows: int | None) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_flow_data(data_dir)
    frame, cleaning = clean_flow_data(frame)

    if max_rows and len(frame) > max_rows:
        # Sample within each label to preserve representation of minority labels.
        sampled = []
        for _, group in frame.groupby("Label", sort=False):
            n = max(1, round(max_rows * len(group) / len(frame)))
            sampled.append(group.sample(n=min(n, len(group)), random_state=42))
        frame = pd.concat(sampled).sample(frac=1, random_state=42)
        if len(frame) > max_rows:
            frame = frame.sample(max_rows, random_state=42)

    y = binary_target(frame["Label"])
    X = frame.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.empty:
        raise ValueError("No usable numeric feature columns found.")
    if len(set(y)) < 2:
        raise ValueError("Both benign and attack examples are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200,
            early_stopping=True, random_state=42
        ),
    }

    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        pipeline = Pipeline([
            ("preprocess", ColumnTransformer(
                [("numeric", numeric_pipeline, list(X_train.columns))],
                remainder="drop"
            )),
            ("model", estimator),
        ])

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        prediction = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics: dict[str, float | str] = {
            "model": name,
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "training_seconds": elapsed,
        }
        rows.append(metrics)

        print(f"\n=== {name} ===")
        print(pd.Series(metrics).to_string())
        print(classification_report(
            y_test, prediction, target_names=["BENIGN", "ATTACK"],
            zero_division=0
        ))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, prediction, display_labels=["BENIGN", "ATTACK"],
            ax=ax, colorbar=False
        )
        ax.set_title(f"{name}: confusion matrix")
        fig.tight_layout()
        fig.savefig(
            output / f"confusion_{name.lower().replace(' ', '_')}.png",
            dpi=160
        )
        plt.close(fig)

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(output / "model_metrics.csv", index=False)
    print("\n=== Model comparison ===")
    print(results.to_string(index=False))

    ax = results.plot(
        x="model", y=["precision", "recall", "f1"],
        kind="bar", figsize=(8, 5), ylim=(0, 1)
    )
    ax.set_ylabel("Score")
    ax.set_title("Model evaluation comparison")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=160)
    plt.close()

    print(f"\nCleaning summary: {cleaning}")
    print(f"Artifacts written to: {output.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--max-rows", type=int, default=200000,
        help="Maximum rows to sample; use 0 for the full dataset."
    )
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.max_rows or None)


if __name__ == "__main__":
    main()
