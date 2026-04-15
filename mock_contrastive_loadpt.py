import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "Difface"))
from faceclip.encoder import Transformer


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
    snps_path = "mock_snps.pt"
    face_latents_path = "mock_face_latents.pt"  #load from the real pt file

    mock_snps(snps_path, num_snps=7842)
    train_snps = load_data(snps_path)

    #mock_eval_latents(face_latents_path, latent_dim=128)
    train_face_latents = load_data(face_latents_path)

    print("[3] Setting up Contrastive Learning with Real Transformer...")

    encoder = Transformer()

    latent_dim = 16
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
