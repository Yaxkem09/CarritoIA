import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2) # (B, embed_dim, n_patches)
        x = x.transpose(1, 2) # (B, n_patches, embed_dim)
        return x

class ModeloBonus(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, num_clases=4, embed_dim=128, depth=4, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(embed_dim, num_clases)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)