import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import sys

sys.path.append(os.path.dirname(__file__))
from encoder import Transformer, CLIP

# ── Data paths ────────────────────────────────────────────────────────────────
SNP_CSV         = os.path.join(os.path.dirname(__file__), 'dataset/mock_snp_processed_800.csv')
FACE_LATENTS_PT = os.path.join(os.path.dirname(__file__), 'dataset/mock_face_latents.pt')

# Decoder inputs — set these tomorrow when raw mesh data is available
MESH_DIR      = None   # TODO: folder containing raw face meshes
TRANSFORM_PKL = None   # TODO: path to transform.pkl (spiral indices, down/up transforms)

USE_DECODER = (MESH_DIR is not None) and (TRANSFORM_PKL is not None)

# ── Model hyperparameters (from main.py) ──────────────────────────────────────
IN_CHANNELS     = 3
OUT_CHANNELS    = [16, 16, 16, 32]
LATENT_CHANNELS = 128
SEQ_LENGTH      = [9, 9, 9, 9]
DILATION        = [1, 1, 1, 1]

# ── Training hyperparameters (from main.py) ───────────────────────────────────
BATCH_SIZE           = 16
EPOCHS               = 200
LR_MODEL             = 0.073e-4
LR_DECODER           = 0.32e-4
WEIGHT_DECAY_MODEL   = 0.30
WEIGHT_DECAY_DECODER = 0.001
LR_DECAY             = 0.99
DECAY_STEP           = 1
TEST_RATIO           = 0.2
SEED                 = 42
DEVICE_IDX           = 0

cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


# ── 1. Load SNP CSV ───────────────────────────────────────────────────────────
# Columns: image_id, geno_id, snp1, ..., snp800  (pre-filtered, no LOG10P row)
print("1. Reading SNP data from CSV...")
image_ids = []
snp_rows  = []

with open(SNP_CSV, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    n_snps = len(header) - 2                              # exclude image_id & geno_id
    for row in reader:
        if not row:
            continue
        image_ids.append(row[0].strip())
        snp_rows.append([float(v) for v in row[2:]])      # skip image_id & geno_id

snp_matrix = np.array(snp_rows, dtype=np.float32)        # (N, n_snps)
print(f"   Loaded: {snp_matrix.shape[0]} subjects × {snp_matrix.shape[1]} SNPs")


# ── 2. Load face data ─────────────────────────────────────────────────────────
print("\n2. Loading face data...")
if USE_DECODER:
    # TODO: load raw face meshes from MESH_DIR aligned to image_ids order
    # face_data should be a tensor of shape (N, num_vertices, 3)
    raise NotImplementedError("Raw mesh loading not yet implemented — set MESH_DIR and TRANSFORM_PKL")
else:
    face_data = torch.load(FACE_LATENTS_PT, weights_only=False)['train']   # (N, FACE_DIM)
    print(f"   Face latents shape: {tuple(face_data.shape)}")
    FACE_DIM = face_data.shape[1]

n_subjects = face_data.shape[0]
assert snp_matrix.shape[0] == n_subjects, (
    f"Subject count mismatch: SNP CSV has {snp_matrix.shape[0]}, face data has {n_subjects}"
)


# ── 3. Build models ───────────────────────────────────────────────────────────
print("\n3. Building models...")
device = torch.device('cuda', DEVICE_IDX) if torch.cuda.is_available() else torch.device('cpu')

if USE_DECODER:
    from encoder import FACE_encoder
    from decoder import Decoder
    import utils

    with open(TRANSFORM_PKL, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    spiral_indices = [
        utils.preprocess_spiral(tmp['face'][i], SEQ_LENGTH[i], tmp['vertices'][i], DILATION[i]).to(device)
        for i in range(len(tmp['face']) - 1)
    ]
    down_transform = [utils.to_sparse(d).to(device) for d in tmp['down_transform']]
    up_transform   = [utils.to_sparse(u).to(device) for u in tmp['up_transform']]

    image_encoder = FACE_encoder(
        IN_CHANNELS, OUT_CHANNELS, LATENT_CHANNELS,
        spiral_indices, down_transform, up_transform,
    ).to(device)
    decoder = Decoder(
        IN_CHANNELS, OUT_CHANNELS, LATENT_CHANNELS,
        spiral_indices, down_transform, up_transform,
    ).to(device)
    print(f"   FACE_encoder + Decoder built")
else:
    image_encoder = nn.Identity()
    decoder       = None
    print(f"   Using nn.Identity() for face (latents mode, no decoder)")

snp_encoder = Transformer(num_snps=n_snps).to(device)
print(f"   Transformer built (num_snps={n_snps})")

model = CLIP(
    image_encoder=image_encoder,
    text_encoder=snp_encoder,
    dim_image=FACE_DIM if not USE_DECODER else LATENT_CHANNELS,
    dim_text=LATENT_CHANNELS,
    dim_latent=LATENT_CHANNELS,
).to(device)
print(f"   CLIP built")


# ── 4. Dataset and loaders ────────────────────────────────────────────────────
print("\n4. Setting up data loaders...")
gene_tensor = torch.tensor(snp_matrix, dtype=torch.float32)
dataset     = TensorDataset(gene_tensor, face_data)
n_test      = int(n_subjects * TEST_RATIO)
n_train     = n_subjects - n_test
train_ds, test_ds = random_split(
    dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
print(f"   Train: {n_train} | Test: {n_test}")


# ── 5. Optimizer and scheduler ────────────────────────────────────────────────
param_groups = [{'params': model.parameters(), 'lr': LR_MODEL, 'weight_decay': WEIGHT_DECAY_MODEL}]
if decoder is not None:
    param_groups.append({'params': decoder.parameters(), 'lr': LR_DECODER, 'weight_decay': WEIGHT_DECAY_DECODER})

optimizer = torch.optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, DECAY_STEP, gamma=LR_DECAY)

out_dir   = os.path.join(os.path.dirname(__file__), 'out', 'checkpoints')
os.makedirs(out_dir, exist_ok=True)


# ── Training functions (following train.py) ───────────────────────────────────
def train(model, decoder, optimizer, loader, device):
    model.train()
    if decoder is not None:
        decoder.train()

    total_loss_clip    = 0.0
    total_loss_decoder = 0.0
    cos_scores = []
    cross = nn.CrossEntropyLoss()

    for snp, face in loader:
        snp, face = snp.to(device), face.to(device)
        optimizer.zero_grad()

        text_features, image_features = model(face, snp)

        # contrastive loss
        logit_scale = model.logit_scale.exp()
        logits1 = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * text_features  @ image_features.t()
        labels  = torch.arange(logits1.size(0)).to(device)
        loss_clip = (cross(logits1, labels) + cross(logits2, labels)) / 2

        cos_scores.append(cos_sim(image_features, text_features))

        if decoder is not None:
            image_embeds  = model.embed_image(face)
            out           = decoder(image_embeds)
            loss_decoder  = F.l1_loss(out, face, reduction='mean')
            loss          = 1 * loss_clip + 10 * loss_decoder
            total_loss_decoder += loss_decoder.item()
        else:
            loss = loss_clip

        loss.backward()
        optimizer.step()
        total_loss_clip += loss_clip.item()

        del snp, face, text_features, image_features
        torch.cuda.empty_cache()

    mean_cos = torch.cat(cos_scores, dim=0).mean().item()
    return total_loss_clip / len(loader), total_loss_decoder / len(loader), mean_cos


@torch.no_grad()
def test(model, decoder, loader, device):
    model.eval()
    if decoder is not None:
        decoder.eval()

    total_loss_clip    = 0.0
    total_loss_decoder = 0.0
    cos_scores = []
    cross = nn.CrossEntropyLoss()

    for snp, face in loader:
        snp, face = snp.to(device), face.to(device)

        text_features, image_features = model(face, snp)

        logit_scale = model.logit_scale.exp()
        logits1 = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * text_features  @ image_features.t()
        labels  = torch.arange(logits1.size(0)).to(device)
        loss_clip = (cross(logits1, labels) + cross(logits2, labels)) / 2

        cos_scores.append(cos_sim(image_features, text_features))

        if decoder is not None:
            image_embeds = model.embed_image(face)
            out          = decoder(image_embeds)
            loss_decoder = F.l1_loss(out, face, reduction='mean')
            total_loss_decoder += loss_decoder.item()

        total_loss_clip += loss_clip.item()

        del snp, face, text_features, image_features
        torch.cuda.empty_cache()

    mean_cos = torch.cat(cos_scores, dim=0).mean().item()
    return total_loss_clip / len(loader), total_loss_decoder / len(loader), mean_cos


def run(model, decoder, train_loader, test_loader, epochs, optimizer, scheduler, device):
    for epoch in range(epochs):
        if epoch % 5 == 0:
            test_clip, test_dec, test_cos = test(model, decoder, test_loader, device)
            print(f"  [Test]  clip={test_clip:.4f}  decoder={test_dec:.4f}  cos={test_cos:.4f}")

        train_clip, train_dec, train_cos = train(model, decoder, optimizer, train_loader, device)
        scheduler.step()
        print(f"  Epoch {epoch:3d}  train_clip={train_clip:.4f}  "
              f"train_decoder={train_dec:.4f}  cos={train_cos:.4f}")

        if epoch % 10 == 0:
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict()}
            if decoder is not None:
                ckpt['decoder_state_dict'] = decoder.state_dict()
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(ckpt, os.path.join(out_dir, f'checkpoint_{epoch:03d}.pt'))
            print(f"  Checkpoint saved → {out_dir}/checkpoint_{epoch:03d}.pt")


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nTraining on {device}  |  USE_DECODER={USE_DECODER}")
run(model, decoder, train_loader, test_loader, EPOCHS, optimizer, scheduler, device)
print("\nDone.")
