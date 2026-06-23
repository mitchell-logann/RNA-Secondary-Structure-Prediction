import RNA
import pandas as pd
import time
import argparse

from pathlib import Path

from data.dataset import BPRNADataset, dotbracketToContact
from evaluation.metrics import *


def runVienna(dataset, numSamples):
    
    results=[]

    experiment_start = time.perf_counter()
    
    for i in range(min(numSamples, len(dataset))):
        sample = dataset[i]
        sequence = sample["sequence_str"]
        
        start_time = time.perf_counter()

        predicted_structure, mfe = RNA.fold(sequence)
        
        runtime = time.perf_counter() - start_time
        
        predicted_contact_map = dotbracketToContact(predicted_structure)

        true_contact_map = sample["contact_map"]

        precision, recall, f1 = evaluateContactMaps(
            true_contact_map, predicted_contact_map
        )
        
        results.append({
            "id": sample["id"],
            "length": sample["length"],
            "mfe": mfe,
            "runtime_sec": runtime,
            "runtime_per_nt": runtime / sample["length"],
            "num_pairs_true": true_contact_map.sum().item() / 2,
            "num_pairs_pred": predicted_contact_map.sum().item() / 2,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
    experiment_runtime = time.perf_counter() - experiment_start
    avg_runtime = experiment_runtime / len(results)
     
    return results, experiment_runtime, avg_runtime
        
        
def saveResults(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ViennaRNA Baseline {len(results)} Results.csv"

    df = pd.DataFrame(results)

    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    return df, csv_path
        
def saveSummary(df, experiment_runtime, avg_runtime, output_dir):
    summary_path = output_dir / f"{len(df)} summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"Vienna Baseline Summary For Sample Size {len(df)}\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Samples Evaluated: {len(df)}\n")
        f.write(f"Total Runtime (sec): {experiment_runtime:.2f}\n")
        f.write(f"Average Runtime per RNA (sec): {avg_runtime:.4f}\n\n")

        f.write(f"Average Precision: {df['precision'].mean():.4f}\n")
        f.write(f"Average Recall:    {df['recall'].mean():.4f}\n")
        f.write(f"Average F1:        {df['f1'].mean():.4f}\n")
        f.write(f"Average MFE:       {df['mfe'].mean():.4f}\n")
        
        

    print(f"Wrote Summary to {summary_path}")
    
def main():
    parser = argparse.ArgumentParser(
        description = "Run ViennaRNA baseline on bpRNA dataset."
    )
    
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of RNA samples to evaluate"
    )
    parser.add_argument(
        "--max_len", type=int, default=None, help="Optional maximum RNA sequence length."
    )
    
    args = parser.parse_args()
    
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles", 
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles",
    )
    
    output_dir = Path(f"./outputs/ViennaRNA Outputs/{args.num_samples}")

    results, experiment_runtime, avg_runtime = runVienna(dataset, args.num_samples)
    
    print(f"Total runtime: {experiment_runtime:.2f} sec")
    print(f"Average per RNA: {avg_runtime:.4f} sec")
    
    df, csv_path = saveResults(results, output_dir)
    
    saveSummary(df, experiment_runtime, avg_runtime, output_dir)
    
if __name__ == "__main__":
    main()