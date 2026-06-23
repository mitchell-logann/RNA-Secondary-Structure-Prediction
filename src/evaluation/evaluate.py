import torch

from src.evaluation.metrics import evaluateContactMaps

@torch.no_grad()
def evaluateModel(model, test_loader, device):
    model.eval()
    
    precision_scores=[]
    recall_scores=[]
    f1_scores=[]
    
    for batch in test_loader:
        sequence = batch["sequence"].to(device)
        targets = batch["contact_map"].to(device)
        logits = model(sequence)
        
        predictions = logitsToContacts(logits)
        
        for i in range(sequence.size(0)):
            precision, recall, f1 = evaluateContactMaps(targets[i], predictions[i])
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
    return {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(recall_scores)
    }
        
def logitsToContacts(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()