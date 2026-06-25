from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(results_path, output_dir, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)

    row = df.iloc[0]

    metrics = ["precision", "recall", "f1"]
    values = [row[m] for m in metrics]

    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{model_name} Test Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_test_metrics.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="CNN")

    args = parser.parse_args()

    plot_results(
        Path(args.results_path),
        Path(args.output_dir),
        args.model_name
    )


if __name__ == "__main__":
    main()