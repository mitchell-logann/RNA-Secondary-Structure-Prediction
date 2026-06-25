from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_training(history_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    if "f1" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["f1"], label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("CNN Validation F1 Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "f1_curve.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    plot_training(
        Path(args.history_path),
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()