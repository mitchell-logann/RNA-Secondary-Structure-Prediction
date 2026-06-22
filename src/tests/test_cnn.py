import torch
from torch.utils.data import DataLoader

from src.models.cnn_contact import CNNContactPredictor
from src.data.dataset import BPRNADataset
from src.data.collate import rna_collate_fn
from src.data.split_dataset import split_dataset

def main():
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles",
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles",
        max_len=512
    )
    
    train_dataset, _, _ = split_dataset(dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=rna_collate_fn
    )
    
    model = CNNContactPredictor()
    batch = next(iter(train_loader))
    sequence = batch["sequence"]
    logits = model(sequence)
    
    print("Sequence shape:", sequence.shape)
    print("Output shape:", logits.shape)
    
if __name__ == "__main__":
    main()