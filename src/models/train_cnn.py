import torch
from torch.utils.data import DataLoader
import argparse

from src.data.dataset import BPRNADataset
from src.data.split_dataset import split_dataset
from src.data.collate import rna_collate_fn
from src.models.cnn_contact import CNNContactPredictor
from src.losses.contact_loss import maskedBCELoss

from pathlib import Path

def trainOneEpoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        sequence = batch["sequence"].to(device)
        targets = batch["contact_map"].to(device)
        mask = batch["mask"].to(device)
        
        logits = model(sequence)
        loss = maskedBCELoss(logits, targets, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

@torch.no_grad()
def validateOneEpoch(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    for batch in val_loader:
        sequence = batch["sequence"].to(device)
        targets = batch["contact_map"].to(device)
        mask = batch["mask"].to(device)
        
        logits = model(sequence)
        loss = maskedBCELoss(logits, targets, mask)
        
        total_loss += loss.item()
        
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(
        description="Train and validate CNN model on bpRNA dataset"
    )
    parser.add_argument(
        "--max_len", type=int, default=None, help="Maximum RNA sequence length."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Number of samples utilized in one epoch."
    )
    parser.add_argument(
        "--epochs", type=int, default = 8, help="Number of iterations the model will run through."
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles", 
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles",
        max_len=args.max_len
    )
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn = rna_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        collate_fn = rna_collate_fn
    )
    
    model = CNNContactPredictor(embed_dim=32,hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    output_dir = Path("outputs/CNN Outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss = trainOneEpoch(model, train_loader, optimizer, device)
        val_loss = validateOneEpoch(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/CNN Outputs/best_cnn.pt")
            print("Saved best model.")
        
if __name__ == "__main__":
    main()