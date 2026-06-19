from src.dataset import BPRNADataset
import matplotlib.pyplot as plt

dataset = BPRNADataset(
    fasta_dir="./bpRNA Dataset/bpRNA_1m/fastaFiles",
    dbn_dir="./bpRNA Dataset/bpRNA_1m/dbnFiles"
)

print("Number of samples:", len(dataset))

# sample = dataset[0]
# 
# print("ID:", sample["id"])
# print("Sequence shape:", sample["sequence"].shape)
# print("Contact map shape:", sample["contact_map"].shape)
# print("Length:", sample["length"])
# print("Number of base pairs:", sample["contact_map"].sum().item() / 2)

for i in [0, 10, 100, 1000]:
    sample = dataset[i]
    
    print(sample["id"])
    print(sample["length"])
    print(sample["contact_map"].shape)
    print(sample["contact_map"].sum().item()/2)


plt.imshow(sample["contact_map"])
plt.title(sample["id"])
plt.colorbar()
plt.show()