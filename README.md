# Difface: De novo reconstruction of 3D human facial images from DNA sequence
# The code for Difface has been released, to use it you need to build your own dataset.

What's working — data pipeline fully verified: 296 subjects × 796 SNPs (LOG10P > 1.5), sorted numerically, vocab 0–47, 237/59 train/test split, mock face latents (296, 16).

Pending — the Transformer flatten → mean-pool change is on hold. The note records both options with rationale so you can pick it up when you're on real hardware.

When you come back on the big dataset, the key things to revisit are:

Apply mean-pool (or CLS token) to encoder.py
Swap mock latents for real FACE_encoder outputs
Tune threshold and batch size for real scale


--------------------

# only modified two existing files:

toServer/Difface/faceclip/contrastive_snp.py — full rewrite with all 7 requirements
toServer/Difface/faceclip/encoder.py — two small changes: try/except guard around torch_scatter/spiralconv imports, and updated the stale comment in Transformer.forward
dataset_260408.py was not touched — it's the full NormalDataset for real face mesh data and isn't needed for the mock contrastive training.
-----------------------------

# Run from the faceclip/ directory using the ControlNet venv:


cd /homes/yusha/Difface/toServer/Difface/faceclip
/homes/yusha/ControlNet/venv/bin/python contrastive_snp.py
That's it — training and evaluation are both inside the same script. It will:

Load and filter the CSV
Sort subjects
Build the model
Save mock_latents.pt
Train for 50 epochs, printing train/test loss every 10 epochs
Save the final model to clip_snp_face.pt in the same directory
Note: it will OOM on the 2 GB dev GPU at the training step (the pending Transformer change). If you want to test the data pipeline only before moving to real hardware, you can add device = torch.device("cpu") temporarily by overriding in the config section at the top of the file.

-----------------------------

# can I say it extracted the contrastive learning part from the main.py
Yes — contrastive_snp.py is essentially the contrastive learning part extracted from main.py, with two substitutions:

FACE_encoder (real mesh encoder requiring spiral transforms, .pkl files, psbody) → replaced with nn.Identity() (passthrough)
Real face latents from NormalDataset → replaced with torch.randn(n_ids, 16)
Everything else — the CLIP model, Transformer, contrastive loss, optimizer — is the same logic. It lets you develop and validate the SNP side independently without needing the full face mesh pipeline.

-------------------------
# version with positional encoding 
Your current file: /homes/yusha/Difface/Difface/faceclip/encoder.py - Missing positional encoder
Version with positional encoder: /homes/yusha/Difface/toServer/Difface/faceclip/encoder.py - Has the positional encoder


-----------------------------
# Option 1: Keep learnable (current - good)
self.pos_encoder = nn.Parameter(torch.randn(1, 1000, embed_dim))

# Option 2: Sinusoidal (traditional transformer - also good)
def get_sinusoidal_pos_encoding(seq_len, d_model):
    positions = torch.arange(seq_len).unsqueeze(1)
    dims = torch.arange(d_model).unsqueeze(0)
    angles = positions / (10000 ** (dims / d_model))
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])
    return pos_encoding.unsqueeze(0)

# Why Option 1 optimal:

SNPs aren't uniformly spaced—some regions have clusters, others are sparse
Learnable encodings capture this irregular distribution
Your training dataset is fixed (~800 SNPs), so no need for length generalization
Lets the model learn that position X might be more important than position Y
Only use sinusoidal if:

You need to handle variable-length SNP sequences at inference time
You want maximum generalization to unseen SNP counts

# 05/05/26 re-write the contrastive learning 

The decoder is a 3D face mesh reconstructor. It takes the 128-dim face latent and reconstructs the original face mesh (3D vertex coordinates) using spiral convolutions on the mesh topology.

Its purpose in the training is as a regularizer — loss2 = F.l1_loss(out, image1) forces the face latent to stay geometrically meaningful during CLIP training, so it doesn't collapse just to minimise the contrastive loss.

For our script it's not applicable for two reasons:

We only have pre-saved face latents (mock_face_latents.pt), not the raw 3D face meshes needed as reconstruction target
The decoder requires mesh transform matrices (spiral indices, down/up transforms from transform.pkl) which we don't have

# Can you leave the need input for the decode as a variable? folder path? tomorrow we will link to read data. I have all the raw meshes and the pt file learnt from spiralnet. pls include both loss?

Done. Here's a summary of the structure:

Tomorrow, to enable the decoder: just set MESH_DIR and TRANSFORM_PKL at the top — USE_DECODER flips to True automatically and the full FACE_encoder + Decoder pipeline activates.

Key design decisions:

train() / test() / run() follow train.py structure exactly
Both losses included: loss = 1 * loss_clip + 10 * loss_decoder (same weights as train.py line 94) — decoder term is skipped when decoder=None
Optimizer with separate LR for model vs decoder, and StepLR scheduler — same as main.py:115-120
Checkpoints saved to out/checkpoints/ with model + decoder + optimizer + scheduler state

-----------------------------
# How to run contrastive_snp.py

## 1. Enabling the decoder (when raw mesh data is available)

Open `Difface/Difface/faceclip/contrastive_snp.py` and set the two path variables near the top:

```python
# Decoder inputs — set these when raw mesh data is available
MESH_DIR      = '/path/to/raw/face/meshes'      # folder containing raw face mesh files
TRANSFORM_PKL = '/path/to/transform.pkl'         # spiral indices, down/up transform matrices
```

Once both are set, `USE_DECODER` becomes `True` automatically and the script will:
- Load `FACE_encoder` and `Decoder` from `encoder.py` / `decoder.py`
- Load mesh transform matrices from `TRANSFORM_PKL`
- Use combined loss: `loss = 1 * loss_clip + 10 * loss_decoder`

When either path is `None`, the script runs in latents-only mode (no decoder, contrastive loss only).

## 2. Bash script to run

```bash
#!/bin/bash
#
# Run contrastive SNP-face training
# Latents-only mode (no decoder): MESH_DIR and TRANSFORM_PKL left as None in script
# To enable decoder: set MESH_DIR and TRANSFORM_PKL in the script first
#

cd /homes/yusha/Difface/Difface/faceclip

/homes/yusha/amd/.env/bin/python contrastive_snp.py
```

Training prints loss every epoch and saves checkpoints every 10 epochs to:
`Difface/Difface/faceclip/out/checkpoints/checkpoint_XXX.pt`

Each checkpoint contains: `model_state_dict`, `decoder_state_dict` (if decoder enabled), `optimizer_state_dict`, `scheduler_state_dict`, `epoch`.

-----------------------------
# Virtual environment: /homes/yusha/amd/.env

## Packages directly used by contrastive_snp.py

| Package   | Version      | Used for                              |
|-----------|--------------|---------------------------------------|
| torch     | 2.7.1+cu118  | models, training, tensors             |
| numpy     | 2.2.6        | SNP CSV loading                       |

CUDA runtime (bundled with torch):
- nvidia-cublas-cu11 11.11.3.6
- nvidia-cudnn-cu11  9.1.0.70
- nvidia-cuda-runtime-cu11 11.8.89

## Full package list

```
absl-py                  2.4.0
contourpy                1.3.2
cycler                   0.12.1
filelock                 3.20.0
fonttools                4.61.1
fsspec                   2025.12.0
grpcio                   1.78.0
Jinja2                   3.1.6
joblib                   1.5.3
kiwisolver               1.4.9
lightning-utilities      0.15.3
Markdown                 3.10.2
MarkupSafe               3.0.2
matplotlib               3.10.8
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.6
nvidia-cublas-cu11       11.11.3.6
nvidia-cuda-cupti-cu11   11.8.87
nvidia-cuda-nvrtc-cu11   11.8.89
nvidia-cuda-runtime-cu11 11.8.89
nvidia-cudnn-cu11        9.1.0.70
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.3.0.86
nvidia-cusolver-cu11     11.4.1.48
nvidia-cusparse-cu11     11.7.5.86
nvidia-nccl-cu11         2.21.5
nvidia-nvtx-cu11         11.8.86
opencv-python            4.13.0.92
packaging                26.0
pillow                   12.0.0
protobuf                 7.34.0
pyparsing                3.3.2
python-dateutil          2.9.0.post0
scikit-learn             1.7.2
scipy                    1.15.3
setuptools               59.6.0
six                      1.17.0
sympy                    1.14.0
tensorboard              2.20.0
tensorboard-data-server  0.7.2
termcolor                3.3.0
threadpoolctl            3.6.0
torch                    2.7.1+cu118
torchaudio               2.7.1+cu118
torchmetrics             1.8.2
torchvision              0.22.1+cu118
tqdm                     4.67.3
triton                   3.3.1
typing_extensions        4.15.0
Werkzeug                 3.1.6
```

Note: `torch_scatter` and `spiralconv` (needed by FACE_encoder and Decoder) are NOT in this env.
They are installed in `/homes/yusha/Difface/venv`. To enable the decoder, either install
`torch_scatter` into this env or switch to an env that has both torch and torch_scatter.