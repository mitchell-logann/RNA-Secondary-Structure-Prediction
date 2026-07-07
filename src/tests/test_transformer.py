import torch
from torch.utils.data import DataLoader

from src.models.transformer_contact import TransformerContactPredictor
from src.data.dataset import BPRNADataset
from src.data.collate import rna_collate_fn
from src.data.split_dataset import split_dataset
from src.losses.contact_loss import maskedBCELoss

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
    
    model = TransformerContactPredictor(
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        max_len=512
    )
    batch = next(iter(train_loader))
    sequence = batch["sequence"]
    targets = batch["contact_map"]
    mask = batch["mask"]
    logits = model(sequence)
    loss = maskedBCELoss(logits, targets, mask)
    
    print("Sequence shape:", sequence.shape)
    print("Output shape:", logits.shape)
    print("Target shape:", targets.shape)
    print("Mask Shape:", mask.shape)
    print("Loss:", loss.item())
    
if __name__ == "__main__":
    main()