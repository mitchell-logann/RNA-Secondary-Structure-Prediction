import torch
import torch.nn as nn

class BiLSTMContact(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.2, padding_idx=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        
        lstm_dim = hidden_dim * 2
        
        self.pair_head = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, 1)
        )
        
    def forward(self, sequence):
        x = self.embedding(sequence)
        h, _ = self.encoder(x)
        B, L, H = h.shape
        
        h_i = h.unsqueeze(2).expand(B, L, L, H)
        h_j = h.unsqueeze(1).expand(B, L, L, H)
        
        pair_features = torch.cat([h_i, h_j], dim=-1)
        
        logits = self.pair_head(pair_features).squeeze(-1)
        logits = (logits + logits.transpose(1, 2)) / 2
        
        return logits