import torch
import time
import pandas as pd
from pathlib import Path

from src.losses.contact_loss import maskedBCELoss
from src.evaluation.evaluate import evaluateModel


def train(model, train_loader, optimizer, device):
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
def validate(model, val_loader, device):
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

def trainModel(model, train_loader, val_loader, test_loader, optimizer, device, output_dir, epochs, checkpoint_name, model_name):
    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True) 
    checkpoint_path = output_dir / checkpoint_name
 
    experiment_start = time.perf_counter()
    history = []
    
    for epoch in range(epochs):
        
        epoch_start = time.perf_counter()
        
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        metrics = evaluateModel(model, val_loader, device)
        

        
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {metrics['f1']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best {model_name} to {checkpoint_path}.")
        
    experiment_runtime = time.perf_counter() - experiment_start
    avg_runtime = experiment_runtime / epochs
    
    history_df = pd.DataFrame(history)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history csv successfully saved to {history_path}")
    
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    
    results = evaluateModel(model, test_loader, device)
    results_df = pd.DataFrame([results])
    results_path = output_dir / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")
    
    summary_path = output_dir / f"{model_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{model_name} Contact Map Predictor Run Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Number of Epochs Ran: {epochs}\n")
        f.write(f"Total Runtime (sec): {experiment_runtime:.2f}\n")
        f.write(f"Average Runtime per Epoch (sec): {avg_runtime:.4f}\n\n")
        
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n\n")
        
        f.write(f"Average Validation Precision: {history_df['precision'].mean():.4f}\n")
        f.write(f"Average Validation Recall:    {history_df['recall'].mean():.4f}\n")
        f.write(f"Average Validation F1:        {history_df['f1'].mean():.4f}\n")
        
        f.write(f"Final Test Metrics From Best Checkpoint\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Precision:    {results['precision']:.4f}\n")
        f.write(f"Test Recall:       {results['recall']:.4f}\n")
        f.write(f"Test F1:           {results['f1']:.4f}\n")
        
    print(f"Summary successfully written to {summary_path}")
    
    return {
        "history": history_df,
        "test_results": results,
        "checkpoint_path": checkpoint_path
    }