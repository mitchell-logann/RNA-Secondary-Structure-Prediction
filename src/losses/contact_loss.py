import torch
import torch.nn.functional as F 

def maskedBCELoss(logits, targets, mask):
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    
    loss = F.binary_cross_entropy_with_logits(
        logits[pair_mask], targets[pair_mask]
    )
    
    return loss