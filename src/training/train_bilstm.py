import torch
from torch.utils.data import DataLoader
import argparse
import time
import pandas as pd

from src.data.dataset import BPRNADataset
from src.data.split_dataset import split_dataset
from src.data.collate import rna_collate_fn
from src.models.bilstm_contact import BiLSTMContact
from src.losses.contact_loss import maskedBCELoss
from src.evaluation.evaluate import evaluateModel
from src.training.trainer import trainModel

from pathlib import Path

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
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn = rna_collate_fn
    )
    
    model = BiLSTMContact(embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    
    trainModel(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        output_dir=Path("outputs/BiLSTM Outputs"),
        epochs=args.epochs,
        checkpoint_name="best_bilstm.pt",
        model_name="BiLSTM"
    )
        
        
if __name__ == "__main__":
    main()