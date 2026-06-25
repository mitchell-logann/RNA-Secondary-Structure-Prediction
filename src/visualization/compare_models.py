from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def compare_models(csv_paths, model_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        row = df.iloc[0]

        rows.append({
            "model": name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"]
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["f1"])
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "model_f1_comparison.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    if len(args.csv_paths) != len(args.model_names):
        raise ValueError("csv_paths and model_names must have the same length.")

    compare_models(
        [Path(p) for p in args.csv_paths],
        args.model_names,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()