import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dataset import BPRNADataset

def visualizeSample(sample,output_dir):
    rnaID = sample["id"]
    contactMap = sample["contact_map"]
    length = sample["length"]
    numPairs = contactMap.sum().item() / 2
    
    plt.figure(figsize=(6,6))
    plt.imshow(contactMap, interpolation="nearest")
    plt.title(f"{rnaID}\nLength: {length}, Base Pairs: {int(numPairs)}")
    plt.xlabel("Nucleotide Position")
    plt.ylabel("Nucleotide Position")
    plt.colorbar(label="Base Pair")
    plt.tight_layout()
    
    outputPath = output_dir / f"{rnaID}_contact_map.png"
    plt.savefig(outputPath, dpi=300)
    plt.close()
    
    print(f"Saved: {outputPath}")
    
def main():
    output_dir = Path("outputs/Contact Maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles", 
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles",
        max_len=512
    )
    
    print("Dataset Size:", len(dataset))
    
    indices = random.sample(range(len(dataset)), 5)
    
    for idx in indices:
        sample = dataset[idx]
        visualizeSample(sample, output_dir)
        
if __name__ == "__main__":
    main()
    

    