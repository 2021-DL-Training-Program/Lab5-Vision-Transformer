import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from transformers import ViTFeatureExtractor

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim = -1)
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.k = nn.Linear(dim, inner_dim, bias=False)
        self.v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        x = self.norm(x)
        b, c, h = x.shape
        q, k, v, = self.q(x).reshape(b, -1, c, self.dim_head), self.k(x).reshape(b, -1, c, self.dim_head), self.v(x).reshape(b, -1, c, self.dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # dot product attention

        attn = self.softmax(dots) # attention weights

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, -1)

        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        self.patch_height, self.patch_width = patch_height, patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print('num patches', num_patches)
        patch_dim = channels * patch_height * patch_width
        print('patch dim', patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def to_patch(self, x):
        batch_size, c, h, w = x.shape
        x = x.reshape(batch_size, -1, self.patch_height*self.patch_width*c) # flatten patches
        return x

    def forward(self, img):
        # x = self.to_patch_embedding(img)
        x = self.to_patch(img)
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)