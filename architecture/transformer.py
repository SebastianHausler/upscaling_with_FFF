import torch
import torch.nn as nn
from einops import rearrange


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., latent_dim=256):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (32 // patch_size) ** 2, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.fc(x)


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., patch_size=4, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (32 // patch_size) ** 2, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.output_proj = nn.Linear(embed_dim, patch_size ** 2 * out_channels)

    def forward(self, x):
        x = self.fc(x).unsqueeze(1)
        x = x + self.pos_embed
        x = self.transformer(x, x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=32 // self.patch_size, w=32 // self.patch_size,
                      p1=self.patch_size, p2=self.patch_size)
        return torch.tanh(x)
