import torch
from torch.utils.data import random_split
from dataset import BPRNADataset

def split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    n = len(dataset)
    
    train_size = int(train_ratio*n)
    val_size = int(val_ratio*n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles", 
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles"
    )
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    print("Total:", len(dataset))
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))