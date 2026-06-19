import torch
from src.models.cnn_contact import CNNContactPredictor

model = CNNContactPredictor()
sequence = torch.randint(0,5, (4,128))
logits = model(sequence)
print(logits.shape)