import torch
from torch.utils.data import DataLoader
import argparse
import pandas as pd

from src.data.dataset import BPRNADataset
from src.data.split_dataset import split_dataset
from src.data.collate import rna_collate_fn
from src.models.bilstm_contact import BiLSTMContact
from src.training.trainer import trainModel
from datetime import datetime

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
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"], help="Optimizer used for training."
    )
    parser.add_argument(
        "--loss_function", type=str, default="Masked BCE", choices=["Masked BCE"], help="Loss function used during training."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5, help="Vocabulary size for the RNA embedding layer."
    )
    parser.add_argument(
        "--padding_idx", type=int, default=4, help="Padding token index."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="Embedding dimension."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for the model."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of encoder layers."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability."
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
    
    model = BiLSTMContact(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 
    
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    config = {
        "model_name": "BiLSTM",
        "dataset": "bpRNA_1m",
        "split_seed": 42,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "loss_function": "Masked BCE",
        "vocab_size": 5,
        "padding_idx": 4,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "device": str(device),
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    
    output_dir = Path("outputs") / "BiLSTM Outputs" / run_id
    
    trainModel(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        checkpoint_name="best_bilstm.pt",
        model_name="BiLSTM",
        config=config
    )
        
        
if __name__ == "__main__":
    main()