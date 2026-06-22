import torch
from torch.utils.data import DataLoader

from src.data.dataset import BPRNADataset
from src.data.split_dataset import split_dataset
from src.data.collate import rna_collate_fn
from src.models.cnn_contact import CNNContactPredictor
from src.losses.contact_loss import maskedBCELoss

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    dataset = BPRNADataset(
        fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles", 
        dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles",
        max_len=96
    )
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        collate_fn = rna_collate_fn
    )
    
    model = CNNContactPredictor(embed_dim=32,hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 1
    
    for epoch in range(num_epochs):
        train_loss = trainOneEpoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}")
        
if __name__ == "__main__":
    main()