import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.function import Function
from sklearn.metrics import classification_report

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum



##Transformer model

class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),       #output_of_attention_dim to mlp_dim
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),      #mlp_dim to output_of_attention_dim(==input_of_attention_dim)
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads       ##qkv_dim(inner_dim) = head_num * head_dim     32 = 4*8
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)   ##embedding_dim to qkv_dim * 3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale #dots for attention-value(scaled) of Q -> K
        
        attn = self.attend(dots)             #atte for Softmax(attention-value)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)        #out for Z = attn * V
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)             #out_dim(==inner_dim==Z_dim) to embedding_dim

class Transformer(nn.Module):           ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x     #Residuals cut-in
            x = ff(x) + x
        return x

class TransModel(nn.Module):                   
    def __init__(self, *, image_size, time_size, fre_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels , dim_head, dropout = 0, emb_dropout = 0.):
        super().__init__()
        assert image_size == 30  ##Time dimensions must equal to 30s
        num_patches = 150       #30*5'
        patch_dim = channels * time_size * fre_size    #4*1*4'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = fre_size, p2 = time_size),
            nn.Linear(patch_dim, dim),        ##patch_dim(16) to embedding_dim
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) ##Generate the pos value'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                  ##Generate the class value'
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
