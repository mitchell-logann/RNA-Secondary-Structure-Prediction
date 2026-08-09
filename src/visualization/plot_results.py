from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_paths, model_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for path, model_name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        row = df.iloc[0]

        rows.append({
            "model": model_name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"]
        })

    comparison_df = pd.DataFrame(rows)

    # Save comparison data
    comparison_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # Plot
    metrics = ["precision", "recall", "f1"]

    x = range(len(comparison_df))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = comparison_df[metric]

        positions = [
            position + (i - 1) * width
            for position in x
        ]

        plt.bar(
            positions,
            values,
            width=width,
            label=metric.capitalize()
        )

    plt.xticks(
        list(x),
        comparison_df["model"]
    )

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Test Performance Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "model_test_metrics_comparison.png"

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Saved comparison CSV to: {output_dir / 'model_comparison.csv'}")
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--csv_paths",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    if len(args.csv_paths) != len(args.model_names):
        raise ValueError(
            "csv_paths and model_names must have the same length."
        )

    plot_results(
        [Path(path) for path in args.csv_paths],
        args.model_names,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
