"""Exploratory analysis for CICIDS2017 flow records."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_utils import clean_flow_data, load_flow_data


def analyse(data_dir: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = load_flow_data(data_dir)
    data, cleaning = clean_flow_data(raw)

    print("\n=== Dataset overview ===")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns):,}")
    print(f"Cleaning summary: {cleaning}")

    print("\n=== Label distribution ===")
    counts = data["Label"].value_counts(dropna=False)
    print(counts.to_string())

    print("\n=== Missing values (top 20) ===")
    missing = data.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(20).to_string() or "No missing values.")

    numeric = data.select_dtypes(include=[np.number])
    infinite = np.isinf(numeric.to_numpy()).sum() if not numeric.empty else 0
    print(f"\nInfinite numeric values: {int(infinite):,}")
    print(f"Duplicate rows after cleaning: {data.duplicated().sum():,}")

    counts.plot(kind="bar", figsize=(12, 5))
    plt.title("CICIDS2017 traffic label distribution")
    plt.xlabel("Traffic label")
    plt.ylabel("Number of flows")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plot_path = output / "label_distribution.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"\nSaved plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    analyse(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
