import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize as norm
from torch_scatter import scatter_add
from spiralconv import SpiralConv

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)
    
def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out

class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class FACE_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform):
        super(FACE_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
       
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x
 
    def forward(self, x, *indices):
        
        z = self.encoder(x)

        return z

class Transformer(nn.Module):
    def __init__(self, num_snps, vocab_size=48):
        super(Transformer, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, 64)  # Embeds 0 to vocab_size-1
        self.pos_encoder = nn.Parameter(torch.randn(1, num_snps, 64))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.layer1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-05)
        self.layer2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-05)

        self.fc2 = nn.Linear(64*num_snps, 128)
        
    def forward(self, x):
        # x is expected to be integers: 0, 1, or 2 representing SNPs
        x = x.long()
        x = self.embedding_layer(x)  # Shape: (batch_size, num_snps, 64)
        x = x + self.pos_encoder     # Add positional embeddings
        
        # PyTorch Transformer expects (Sequence_Length, Batch_Size, Embedding_Dim)
        x = x.permute(1, 0, 2)       # Shape: (num_snps, batch_size, 64)
        
        x1 = self.layer1(x) 
        x1 = x1 + x
        x2 = self.layer2(x1)
        x2 = x2 + x1
        
        # Bring it back to (Batch_Size, Sequence_Length, Embedding_Dim)
        x2 = x2.permute(1, 0, 2)
        x2 = torch.flatten(x2, start_dim = 1) # Shape: (batch_size, num_snps * 64)
        out = self.relu2(self.fc2(x2))

        return out

class CLIP(nn.Module):
    
    def __init__(self, image_encoder, text_encoder, dim_text = 128,
        dim_image = 128,
        dim_latent = 128):
        super().__init__()

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent
        self.encoder1 = image_encoder
        self.encoder2 = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # text latent projection
        self.to_text_latent = nn.Linear(dim_text, dim_latent)
        # image latent projection
        self.to_image_latent = nn.Linear(dim_image, dim_latent)

    def forward(self, image, text):

        image_embeds = self.encoder1(image)
        image_features = self.to_image_latent(image_embeds)

        text_embeds = self.encoder2(text)
        text_features = self.to_text_latent(text_embeds)

        text_latents, image_latents = map(l2norm, (text_features, image_features))

        return  text_latents, image_latents
    
    def embed_text(self,text):
        text_embeds = self.encoder2(text)
        text_features = self.to_text_latent(text_embeds)

        return text_features
    
    def embed_image(self,image):
        image_embeds = self.encoder1(image)
        image_features = self.to_image_latent(image_embeds)

        return image_features




        
    
 
