from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_training(history_paths, model_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)


    # -------------------------
    # Loss comparison
    # -------------------------

    plt.figure(figsize=(10, 6))

    for path, model_name in zip(history_paths, model_names):
        df = pd.read_csv(path)

        plt.plot(
            df["epoch"],
            df["train_loss"],
            label=f"{model_name} Train"
        )

        plt.plot(
            df["epoch"],
            df["val_loss"],
            linestyle="--",
            label=f"{model_name} Validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = output_dir / "model_loss_comparison.png"

    plt.savefig(
        loss_path,
        dpi=300
    )

    plt.close()

    # -------------------------
    # F1 comparison
    # -------------------------

    plt.figure(figsize=(10, 6))

    for path, model_name in zip(history_paths, model_names):
        df = pd.read_csv(path)

        if "f1" in df.columns:
            plt.plot(
                df["epoch"],
                df["f1"],
                label=model_name
            )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.title("Validation F1 Across Epochs")
    plt.legend()
    plt.tight_layout()

    f1_path = output_dir / "model_f1_comparison.png"

    plt.savefig(
        f1_path,
        dpi=300
    )

    plt.close()

    print(f"Saved loss comparison to: {loss_path}")
    print(f"Saved F1 comparison to: {f1_path}")


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--history_paths",
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

    if len(args.history_paths) != len(args.model_names):
        raise ValueError(
            "history_paths and model_names must have the same length."
        )

    plot_training(
        [Path(path) for path in args.history_paths],
        args.model_names,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
