import torch

def rna_collate_fn(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()
    batch_size = len(batch)
    
    sequences = torch.full((batch_size, max_len), fill_value=4, dtype=torch.long)
    contacts = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    masks = torch.zeros((batch_size,max_len),dtype=torch.bool)
    
    ids = []
    
    for i, item in enumerate(batch):
        L = item["length"]
        
        sequences[i, :L] = item["sequence"]
        contacts[i, :L, :L] = item["contact_map"]
        masks[i, :L] = True
        ids.append(item["id"])
        
    return {
        "sequence": sequences,
        "contact_map": contacts,
        "mask": masks,
        "length": lengths,
        "id": ids
    }