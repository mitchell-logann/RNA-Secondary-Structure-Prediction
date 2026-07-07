import torch
import torch.nn as nn

class TransformerContactPredictor(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_len=1024, padding_idx=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=4)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = hidden_dim,
            dropout = dropout,
            batch_first = True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pair_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sequence):
        B, L = sequence.shape
        
        positions = torch.arange(L, device=sequence.device).unsqueeze(0).expand(B,L)
        
        x = self.embedding(sequence) + self.pos_embedding(positions)
        
        padding_mask = sequence.eq(4)
        
        h = self.transformer(x, src_key_padding_mask=padding_mask)

        H = h.shape[-1]
        
        h_i = h.unsqueeze(2).expand(B, L, L, H)
        h_j = h.unsqueeze(1).expand(B, L, L, H)
        
        pair_features = torch.cat([h_i,h_j], dim=-1)
        
        logits = self.pair_head(pair_features).squeeze(-1)
        logits = (logits + logits.transpose(1,2)) / 2
        
        return logits