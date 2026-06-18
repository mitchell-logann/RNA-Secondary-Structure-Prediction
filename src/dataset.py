import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from Bio import SeqIO

VOCAB = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "N": 4 
}
PAIR_OPEN = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">"
}
PAIR_CLOSE = {v: k for k, v in PAIR_OPEN.items()}

class BPRNADataset(Dataset):
    def __init__(self,fasta_dir,dbn_dir,max_len=None):
        self.samples = []
        
        for fasta_file in Path(fasta_dir).glob("*.fasta"):
            stem = fasta_file.stem
            dbn_file = Path(dbn_dir) / f"{stem}.dbn"
            if dbn_file.exists():
                if max_len is not None:
                    sequence = loadFasta(fasta_file)
                    if len(sequence) > max_len:
                        continue   
                self.samples.append((fasta_file,dbn_file))

    def __len__(self):
        return len(self.samples)   
    
    def __getitem__(self, idx):
        fasta_file, dbn_file = self.samples[idx]
        
        sequence = loadFasta(fasta_file)
        structure = loadDbn(dbn_file)
        encoded_seq = encodeSequence(sequence)
        contact_map = dotbracketToContact(structure)
        
        return {
            "sequence": encoded_seq,
            "sequence_str": sequence,
            "contact_map": contact_map,
            "length": len(sequence),
            "id": fasta_file.stem
        }

def loadFasta(fasta_file):
    with open(fasta_file, "r") as handle:
        record = next(SeqIO.parse(handle,"fasta"))
        return str(record.seq).upper().replace("T", "U")
        
def loadDbn(dbn_file):
    with open(dbn_file, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
        
    if len(lines) < 2:
        raise ValueError(f"DBN file does not contain sequence and structure: {dbn_file}")
    
    sequence = lines[0]
    structure = lines[1]
    
    if len(sequence) != len(structure):
        raise ValueError(
            f"Sequence and structure lengths do not match in {dbn_file}: "
            f"{len(sequence)} vs {len(structure)}"
        )
    
    return structure

def encodeSequence(sequence):
    sequence = sequence.upper().replace("T","U")
    return torch.tensor([VOCAB.get(base, 4) for base in sequence], dtype = torch.long)


def dotbracketToContact(structure):
    L = len(structure)
    contact = torch.zeros((L,L), dtype=torch.float32)
    stacks = {open_char: [] for open_char in PAIR_OPEN}
    
    for i, char in enumerate(structure):
        if char in PAIR_OPEN:
            stacks[char].append(i)
            
        elif char in PAIR_CLOSE:
            open_char = PAIR_CLOSE[char]
            if stacks[open_char]:
                j = stacks[open_char].pop()
                contact[i,j] = 1.0
                contact[j,i] = 1.0
    
    return contact
