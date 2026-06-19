from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

results_path = Path("outputs/ViennaRNA Outputs/1000/ViennaRNA Baseline 1000 Results.csv")
output_dir = Path("outputs/ViennaRNA Outputs/1000")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_path)

# 1. F1 distribution
plt.figure(figsize=(8, 5))
plt.hist(df["f1"], bins=30)
plt.xlabel("F1 Score")
plt.ylabel("Number of RNAs")
plt.title("ViennaRNA F1 Score Distribution")
plt.tight_layout()
plt.savefig(output_dir / "f1_distribution.png", dpi=300)
plt.close()

# 2. F1 vs sequence length
plt.figure(figsize=(8, 5))
plt.scatter(df["length"], df["f1"], alpha=0.6)
plt.xlabel("Sequence Length")
plt.ylabel("F1 Score")
plt.title("ViennaRNA F1 Score vs Sequence Length")
plt.tight_layout()
plt.savefig(output_dir / "f1_vs_length.png", dpi=300)
plt.close()

# 3. Runtime vs sequence length
plt.figure(figsize=(8, 5))
plt.scatter(df["length"], df["runtime_sec"], alpha=0.6)
plt.xlabel("Sequence Length")
plt.ylabel("Runtime (seconds)")
plt.title("ViennaRNA Runtime vs Sequence Length")
plt.tight_layout()
plt.savefig(output_dir / "runtime_vs_length.png", dpi=300)
plt.close()

# 4. Precision vs recall
plt.figure(figsize=(8, 5))
plt.scatter(df["recall"], df["precision"], alpha=0.6)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("ViennaRNA Precision vs Recall")
plt.tight_layout()
plt.savefig(output_dir / "precision_vs_recall.png", dpi=300)
plt.close()

print("Saved plots to:", output_dir)