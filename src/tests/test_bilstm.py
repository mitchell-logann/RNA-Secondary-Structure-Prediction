from torch.utils.data import DataLoader

from src.data.dataset import BPRNADataset
from src.data.split_dataset import split_dataset
from src.data.collate import rna_collate_fn
from src.models.bilstm_contact import BiLSTMContact
from src.losses.contact_loss import maskedBCELoss
from pathlib import Path



def main():
    
    fasta_dir = Path("./bpRNA Dataset/bpRNA_1m/fastaFiles")
    dbn_dir = Path("./bpRNA Dataset/bpRNA_1m/dbnFiles")

    print("FASTA exists:", fasta_dir.exists())
    print("DBN exists:", dbn_dir.exists())
    print("FASTA files:", len(list(fasta_dir.glob("*.fasta"))))
    print("DBN files:", len(list(dbn_dir.glob("*.dbn"))))
    
    dataset = BPRNADataset(
        fasta_dir=fasta_dir,
        dbn_dir=dbn_dir,
        max_len=512
    )
    
    print("Dataset size:", len(dataset))
    train_dataset, _, _ = split_dataset(dataset)
    print("Train size:", len(train_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=rna_collate_fn)
    model = BiLSTMContact(embed_dim=32,hidden_dim=64,num_layers=1)
    
    batch = next(iter(train_loader))
    
    sequence = batch["sequence"]
    targets = batch["contact_map"]
    mask = batch["mask"]
    
    logits = model(sequence)
    loss = maskedBCELoss(logits, targets, mask)
    
    print("Sequence shape:", sequence.shape)
    print("Output shape:", logits.shape)
    print("Target shape:", targets.shape)
    print("Mask shape:", mask.shape)
    print("Loss:", loss.item())
    
if __name__ == "__main__":
    main()