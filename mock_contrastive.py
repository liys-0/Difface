import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------
# 1. Minimal Encoders (replacing the complex 3D/Transformer ones)
# ---------------------------------------------------------
class MockDNAEncoder(nn.Module):
    def __init__(self, input_dim=1000, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MockFaceEncoder(nn.Module):
    def __init__(self, input_dim=5000, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. The CLIP-style Contrastive Model
# ---------------------------------------------------------
def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class MockCLIP(nn.Module):
    def __init__(
        self, image_encoder, text_encoder, dim_text=128, dim_image=128, dim_latent=128
    ):
        super().__init__()
        self.encoder1 = image_encoder  # Face
        self.encoder2 = text_encoder  # DNA
        self.logit_scale = nn.Parameter(torch.ones([1]))

        self.to_text_latent = nn.Linear(dim_text, dim_latent)
        self.to_image_latent = nn.Linear(dim_image, dim_latent)

    def forward(self, image, text):
        image_embeds = self.encoder1(image)
        image_features = self.to_image_latent(image_embeds)

        text_embeds = self.encoder2(text)
        text_features = self.to_text_latent(text_embeds)

        # L2 Normalize before dot product
        text_latents = l2norm(text_features)
        image_latents = l2norm(image_features)

        return text_latents, image_latents


# ---------------------------------------------------------
# 3. Training Loop with Mock Data
# ---------------------------------------------------------
def main():
    # Hyperparameters
    batch_size = 32
    num_samples = 320
    dna_dim = 1000
    face_dim = 5000
    epochs = 100

    print("Generating mock data...")
    # Mock Data: DNA sequences (random noise)
    mock_dna = torch.randn(num_samples, dna_dim)

    # Mock Data: Faces. We add a weak correlation to the DNA so the model can actually learn a mapping.
    # In reality, this would be your 3D mesh vertices or extracted features.
    mock_face = (
        torch.randn(num_samples, face_dim) + mock_dna.mean(dim=1, keepdim=True) * 0.5
    )

    dataset = TensorDataset(mock_dna, mock_face)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MockCLIP(MockFaceEncoder(face_dim), MockDNAEncoder(dna_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cross_entropy = nn.CrossEntropyLoss()

    print("Starting contrastive training...")
    for epoch in range(epochs):
        total_loss = 0
        for dna_batch, face_batch in dataloader:
            optimizer.zero_grad()

            # Forward pass
            text_features, image_features = model(face_batch, dna_batch)

            # Contrastive Loss Calculation (InfoNCE)
            logit_scale = model.logit_scale.exp()
            logits1 = logit_scale * image_features @ text_features.t()
            logits2 = logit_scale * text_features @ image_features.t()
            labels = torch.arange(logits1.size(0))
            cross = nn.CrossEntropyLoss()
            loss_i = cross(logits1, labels)
            loss_t = cross(logits2, labels)

            # Average the symmetric loss
            loss1 = (loss_i + loss_t) / 2

            loss1.backward()
            optimizer.step()

            total_loss += loss1.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}] | Contrastive Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
