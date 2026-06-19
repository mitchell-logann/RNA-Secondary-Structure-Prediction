import torch
import torch.nn as nn 

class CNNContactPredictor(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=4)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sequence):
        x = self.embedding(sequence)
        x = x.transpose(1, 2)
        
        h = self.encoder(x)
        h = h.transpose(1,2)
        
        B, L, H = h.shape
        
        h_i = h.unsqueeze(2).expand(B, L, L, H)
        h_j = h.unsqueeze(1).expand(B, L, L, H)
        
        pair_features = torch.cat([h_i,h_j], dim=-1)
        
        logits = self.pair_head(pair_features).squeeze(-1)
        logits = (logits + logits.transpose(1,2)) / 2
        
        return logits