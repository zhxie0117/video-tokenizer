import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.model_new.base.rope import apply_rotary_emb

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attn(nn.Module):
    def __init__(self, dim, heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs):
        B, N, C = x.shape
        
        # === 修改处开始 ===
        # 之前的错误写法导致 Dim 1 是 Heads (12)，无法与 freqs (2048) 广播
        # 正确写法：保持 Dim 1 为 Sequence Length (N)
        # [B, N, 3*C] -> [B, N, 3, Heads, Dim] -> [3, B, N, Heads, Dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        
        # 此时 q, k 的形状是 [Batch, N(Seq), Heads, Dim]
        # freqs 的形状通常是 [1, N(Seq), 1, Dim] 或 [N(Seq), Dim]
        # 现在 Dim 1 都是 N (2048)，可以正确应用 RoPE
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        # PyTorch 的 scaled_dot_product_attention 需要 [Batch, Heads, Seq, Dim]
        # 所以在这里进行转置
        q = q.transpose(1, 2) # [B, Heads, N, Dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # === 修改处结束 ===

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.
        )

        # [B, Heads, N, Dim] -> [B, N, Heads*Dim]
        x = x.transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attn(dim, heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.mlp(self.norm2(x))
        return x

class ResidualAttentionBlock1(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            heads=8,
            mlp_ratio=4,
            num_layer=2,
            drop=0.,
            attn_drop=0.
        ): 
        super().__init__()
        self.num_layer = num_layer
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=False, 
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(num_layer)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, freqs):
        for block in self.blocks:
            x = block(x, freqs)
            
        x = self.norm(x)
        return x