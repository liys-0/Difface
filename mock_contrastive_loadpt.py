import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

import csv
import numpy as np
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "Difface"))
from faceclip.encoder import Transformer


def map_categories_to_012(X):
    X_mapped = np.zeros_like(X, dtype=np.int32)
    for i in range(1, 49):
        X_mapped[X == i] = i % 3
    return X_mapped


def load_category_csv_to_ram(csv_path: str, dtype=np.int32):
    ids: List[str] = []
    rows: List[np.ndarray] = []
    log10p = None

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        snp_cols = header[1:]
        for row in r:
            if len(row) == 0:
                continue
            row_name = row[0].strip()
            if row_name.upper() == "LOG10P":
                log10p = np.asarray(row[1:], dtype=np.float32)
                continue
            ids.append(row_name)
            rows.append(np.asarray(row[1:], dtype=dtype))

    X = np.stack(rows, axis=0)
    X = map_categories_to_012(X)
    id2row: Dict[str, int] = {pid: i for i, pid in enumerate(ids)}
    return ids, X, id2row, snp_cols, log10p


def mock_eval_latents(pt_path, num_train=1000, latent_dim=128):
    print(f"[1] Mocking face latent vectors to '{pt_path}'...")
    train_latents = torch.randn(num_train, latent_dim)

    out_dict = {"train": train_latents}
    torch.save(out_dict, pt_path)
    print(f"    Saved {num_train} train latents (dim={latent_dim}).\n")


def mock_snps(pt_path, num_train=256, num_snps=7842):
    print(f"[1] Mocking SNP data to '{pt_path}'...")
    train_snps = torch.randint(0, 3, (num_train, num_snps)).float()

    out_dict = {"train": train_snps}
    torch.save(out_dict, pt_path)
    print(f"    Saved {num_train} train SNPs (num_snps={num_snps}).\n")


def load_data(pt_path):
    print(f"[2] Loading data from '{pt_path}'...")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File {pt_path} not found.")

    data = torch.load(pt_path)
    train_data = data["train"]

    print(f"    Successfully loaded train data: {train_data.shape}\n")
    return train_data


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    z = F.normalize(z, dim=-1)
    sim_matrix = torch.matmul(z, z.T) / temperature

    sim_ij = torch.diag(sim_matrix, batch_size)
    sim_ji = torch.diag(sim_matrix, -batch_size)

    positives = torch.cat([sim_ij, sim_ji], dim=0)
    nominator = torch.exp(positives)

    mask = (
        (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool))
        .float()
        .to(z.device)
    )
    denominator = mask * torch.exp(sim_matrix)

    loss = -torch.log(nominator / denominator.sum(dim=1))
    return loss.mean()


def main():
    csv_path = "Difface/faceclip/dataset/mock_snp_1000_category_ids_mapped.csv"
    print(f"[1] Loading SNPs from '{csv_path}'...")
    ids, snp_matrix, id2row, snp_cols, log10p = load_category_csv_to_ram(csv_path)

    # Sort subjects by ascending ID
    ids_sorted = sorted(ids)
    row_order = [id2row[i] for i in ids_sorted]
    snp_matrix = snp_matrix[row_order]
    ids = ids_sorted

    # Filter SNPs by LOG10P threshold (~800/1000 kept)
    log10p_threshold = 1.49
    snp_keep_mask = log10p > log10p_threshold
    snp_matrix = snp_matrix[:, snp_keep_mask]

    num_train, num_snps = snp_matrix.shape
    train_snps = torch.tensor(snp_matrix).float()
    print(f"    Loaded {num_train} subjects, {num_snps} SNPs kept (LOG10P > {log10p_threshold}).\n")

    face_latents_path = "mock_face_latents.pt"
    latent_dim = 16
    mock_eval_latents(face_latents_path, num_train=num_train, latent_dim=latent_dim)
    train_face_latents = load_data(face_latents_path)

    print("[3] Setting up Contrastive Learning with Real Transformer...")

    encoder = Transformer(num_snps=num_snps)

    model_snps = ProjectionHead(in_dim=128, out_dim=8)
    model_faces = ProjectionHead(in_dim=latent_dim, out_dim=8)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(model_snps.parameters()) + list(model_faces.parameters()), lr=1e-3
    )

    dataset = TensorDataset(train_snps, train_face_latents)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    epochs = 5
    print(f"    Starting training for {epochs} epochs...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    model_snps = model_snps.to(device)
    model_faces = model_faces.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_snps, batch_faces in loader:
            batch_snps = batch_snps.to(device)
            batch_faces = batch_faces.to(device)

            optimizer.zero_grad()

            # Encode SNPs (using the real Transformer)
            z_snps = encoder(batch_snps)

            # Project both modalities to the same shared space (dim=8 here)
            proj_snps = model_snps(z_snps)
            proj_faces = model_faces(batch_faces)

            # Contrastive loss between SNPs and Face Latents
            loss = nt_xent_loss(proj_snps, proj_faces, temperature=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"    Epoch [{epoch + 1}/{epochs}] - Contrastive Loss: {avg_loss:.4f}")

    print("\n[4] Training complete!")


if __name__ == "__main__":
    main()
