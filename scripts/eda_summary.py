# Adapted from coffeefuelbump/csv-data-summarizer-claude-skill (analyze.py),
# stripped of that repo's SKILL.md behavioral directives — kept only the
# pandas/matplotlib analysis logic, generalized for the global warming dataset.
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_csv(file_path, out_dir="."):
    df = pd.read_csv(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    charts_created = []

    summary.append("=" * 60)
    summary.append("DATA OVERVIEW")
    summary.append("=" * 60)
    summary.append(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")

    summary.append("\nDATA TYPES:")
    for col, dtype in df.dtypes.items():
        summary.append(f"  - {col}: {dtype}")

    missing = df.isnull().sum().sum()
    missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
    summary.append("\nDATA QUALITY:")
    if missing:
        summary.append(f"Missing values: {missing:,} ({missing_pct:.2f}% of total data)")
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            if col_missing > 0:
                col_pct = (col_missing / len(df)) * 100
                summary.append(f"  - {col}: {col_missing:,} ({col_pct:.1f}%)")
    else:
        summary.append("No missing values.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary.append("\nNUMERICAL ANALYSIS:")
        summary.append(str(df[numeric_cols].describe()))

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            plt.figure(figsize=(14, 12))
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, square=True, linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            path = out_dir / "correlation_heatmap.png"
            plt.savefig(path, dpi=150)
            plt.close()
            charts_created.append(str(path))

    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if "id" not in c.lower()]
    if categorical_cols:
        summary.append("\nCATEGORICAL ANALYSIS:")
        for col in categorical_cols[:5]:
            value_counts = df[col].value_counts()
            summary.append(f"\n{col}: {df[col].nunique()} unique values")
            for val, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                summary.append(f"  - {val}: {count:,} ({pct:.1f}%)")

    if charts_created:
        summary.append("\nVISUALIZATIONS CREATED:")
        for chart in charts_created:
            summary.append(f"  - {chart}")

    return "\n".join(summary)


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "global_warming_dataset.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "eda_output"
    print(summarize_csv(file_path, out_dir))
