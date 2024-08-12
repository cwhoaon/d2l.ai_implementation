import torch
import torch.nn as nn
from Attention import MultiHeadAttention

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, X):
        # X: (batch_size, num_channels, img_X, img_Y)
        # output: (batch_size, num_patches, num_hiddens)
        return self.conv(X).flatten(start_dim=2, end_dim=3).transpose(1, 2)
    
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        X = self.dropout1(self.gelu(self.dense1(X)))
        return self.dropout2(self.dense2(X))

class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
    
    def forward(self, X, valid_lens=None):
        Y = self.ln1(X)
        X = self.attention(Y, Y, Y, valid_lens) + X
        return self.mlp(self.ln2(X)) + X

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout, use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches+1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential(*[ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens, num_heads, blk_dropout, use_bias) for _ in range(num_blks)])
        self.head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )
    
    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), dim=1)
        X = self.blks(X + self.pos_embedding)
        return self.head(X[:, 0])




