"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack

import random

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.dim = default(dim, len(_levels))

        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        bounded = self.bound(z)
        quantized = round_ste(bounded)
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        # assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)
        codes = self._indices_to_codes(indices)
        return codes
    
    @torch.compiler.disable()
    def forward(self, z): # (B*L)C in

        with torch.autocast(z.device.type, enabled=False):
            orig_dtype = z.dtype
            z = z.float()

            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)

            codes = codes.to(orig_dtype)

        return codes, {'indices': indices}
    







from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, input_format='bchw', predefined_codebook='/data2/zhxie/myproject/bsq-vit/cache/leech_lattices_normalized.npy', freeze_codebook=True):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ['bchw', 'blc']
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)
        self.bits_per_index = int(np.ceil(np.log2(n_embed)))

        if predefined_codebook is not None:
            predefined_codebook = torch.from_numpy(np.load(predefined_codebook))
            assert predefined_codebook.shape == (n_embed, embed_dim), 'Predefined codebook has incorrect shape'
            self.embedding.weight.data.copy_(predefined_codebook)
            if freeze_codebook:
                print(f"Freezing codebook weights. {self.embedding.weight.shape=}")
                self.embedding.weight.requires_grad = False
            else:
                print(f"Initializing the codebook from {predefined_codebook}, and the codebook is trainable.")
                self.embedding.weight.requires_grad = True

    def forward(self, z):
        batch = z.shape[0]
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')

        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = F.normalize(self.embedding.weight, dim=-1)
            d = -z_flatten @ embedding_weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flatten @ self.embedding.weight.t()

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = torch.bincount(min_encoding_indices.long(), minlength=self.n_embed).float()
        cb_entropy = self.get_entropy(cb_usage)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        # fix the issue with loss scaling
        # loss weight should not associate with the dimensionality of words
        # loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta * torch.mean(((z_q.detach() - z) ** 2).sum(dim=-1)) + torch.mean(((z_q - z.detach()) ** 2).sum(dim=-1))

        z_q = z + (z_q - z).detach()
        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        returndict={'output': z_q, 'loss_codebook': loss}
        return returndict

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        H = -(probs * torch.log(probs)).sum()
        return H


    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        if self.input_format == 'bchw':
            h = w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q