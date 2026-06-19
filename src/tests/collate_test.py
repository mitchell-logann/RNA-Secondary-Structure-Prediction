from torch.utils.data import DataLoader
from src.dataset import BPRNADataset
from src.split_dataset import split_dataset
from src.collate import rna_collate_fn

dataset = BPRNADataset(
    fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles",
    dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles"
)

train_dataset, val_dataset, test_dataset = split_dataset(dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=rna_collate_fn
)

batch = next(iter(train_loader))

print(batch["sequence"].shape)      # [B, L]
print(batch["contact_map"].shape)   # [B, L, L]
print(batch["mask"].shape)          # [B, L]
print(batch["length"])