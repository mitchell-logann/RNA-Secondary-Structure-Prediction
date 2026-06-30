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
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn = rna_collate_fn
    )
    
    model = BiLSTMContact(embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    output_dir = Path("outputs/BiLSTM Outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_start = time.perf_counter()
    history = []
    
    for epoch in range(args.epochs):
        
        epoch_start = time.perf_counter()
        
        train_loss = trainOneEpoch(model, train_loader, optimizer, device)
        val_loss = validateOneEpoch(model, val_loader, device)
        metrics = evaluateModel(model, val_loader, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/BiLSTM Outputs/best_bilstm.pt")
            print("Saved best model.")
        
        epoch_runtime = time.perf_counter() - epoch_start
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "runtime_sec": epoch_runtime
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {metrics['f1']:.4f}")
            
    experiment_runtime = time.perf_counter() - experiment_start
    avg_runtime = experiment_runtime / args.epochs
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    print("Results csv successfully saved")
    
    model.load_state_dict(
        torch.load("outputs/BiLSTM Outputs/best_bilstm.pt", map_location=device)
    )
    
    results = evaluateModel(model, test_loader, device)
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    
    summary_path = output_dir / "BiLSTM_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"CNN Contact Map Predictor Run Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Number of Epochs Ran: {args.epochs}\n")
        f.write(f"Total Runtime (sec): {experiment_runtime:.2f}\n")
        f.write(f"Average Runtime per RNA (sec): {avg_runtime:.4f}\n\n")
        
        f.write(f"Average Precision: {history_df['precision'].mean():.4f}\n")
        f.write(f"Average Recall:    {history_df['recall'].mean():.4f}\n")
        f.write(f"Average F1:        {history_df['f1'].mean():.4f}\n")
        
        f.write(f"Best Precision:    {results['precision']:.4f}\n")
        f.write(f"Best Recall:       {results['recall']:.4f}\n")
        f.write(f"Best F1:           {results['f1']:.4f}\n")
        
    print(f"Summary successfully written to {summary_path}")    
        
        
if __name__ == "__main__":
    main()