import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from encoder import Transformer, CLIP

print("1. Reading data from CSV...")
csv_file = './dataset/mock_snp_processed_800.csv'

row_img_ids = []
row_geno_ids = []
data_matrix = []

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # Row 0: image_id, geno_id, chr1:...
    
    for i, row in enumerate(reader):
        if not row:
            continue
        row_img_ids.append((i, row[0])) # Store original index and image_id
        row_geno_ids.append(row[1])
        data_matrix.append([float(x) for x in row[2:]])

data_matrix = np.array(data_matrix)
print(f"Original data shape: {data_matrix.shape}")

filtered_data = data_matrix

# 2. Sort the first col IDs (rows) in ascending order
print("2. Sorting rows by first col IDs in ascending order...")

# Helper to sort IDs numerically if possible, otherwise string sort
def get_sort_key(k):
    val = row_img_ids[k][1]
    try:
        return (0, int(''.join(filter(str.isdigit, val))))
    except ValueError:
        return (1, val)

# Sort row_img_ids by the ID string in ascending order
sorted_indices = sorted(range(len(row_img_ids)), key=get_sort_key, reverse=False)

filtered_data = filtered_data[sorted_indices]
sorted_row_ids = [row_img_ids[i][1] for i in sorted_indices]

# Convert data to tensor
filtered_data[filtered_data == -1] = 0
# The data values are categorical (integers mapped to float in CSV)
gene_tensor = torch.tensor(filtered_data, dtype=torch.long)
n_ids, n_genes = gene_tensor.shape
print(f"Filtered and sorted tensor shape: {gene_tensor.shape} (N_ids: {n_ids}, N_genes: {n_genes})")

# 3. Transformer model to get N_ids * 128 dimension
print("3. Building Transformer model...")

vocab_size = int(gene_tensor.max().item()) + 1
print(f"Max category ID (vocab size): {vocab_size}")

text_encoder = Transformer(num_snps=n_genes, vocab_size=vocab_size)

image_encoder = nn.Identity()

model = CLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_text=128,
    dim_image=16,
    dim_latent=128
)

# 4. Mock a latent vector in pt format (N_ids * 16)
print("4. Mocking latent vectors...")
mock_latents = torch.randn(n_ids, 16)
torch.save(mock_latents, 'mock_latents.pt')
print(f"Saved mock_latents.pt with shape {mock_latents.shape}")

# 5. Train the contrastive learning
print("5. Training contrastive learning...")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = TensorDataset(gene_tensor, mock_latents)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

cross = nn.CrossEntropyLoss()
epochs = 50
model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for batch_snps, batch_faces in loader:
        batch_snps = batch_snps.to(device)
        batch_faces = batch_faces.to(device)

        optimizer.zero_grad()

        text_features, image_features = model(batch_faces, batch_snps)

        logit_scale = model.logit_scale.exp()
        logits1 = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * text_features @ image_features.t()

        labels = torch.arange(logits1.size(0), device=device)
        loss_i = cross(logits1, labels)
        loss_t = cross(logits2, labels)
        loss = (loss_i + loss_t) / 2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Contrastive Loss: {avg_loss:.4f}")

print("Training completed successfully!")
