import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import itertools
from einops import rearrange
import einops
from easydict import EasyDict as edict

import itertools
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from easydict import EasyDict as edict

import models
import utils
from models import register

from .embed import (PatchEmbed3D, VideoPatchEmbed,
                    get_1d_sincos_pos_embed_from_grid, get_3d_sincos_pos_embed)
from vjepa2.src.models.vision_transformer import vit_large_rope
import vjepa2.src.datasets.utils.video.transforms as video_transforms
import vjepa2.src.datasets.utils.video.volume_transforms as volume_transforms
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# =========================================================
# 1) 工具：下载（可选）
# =========================================================
def download_file(url, local_path, chunk_size=1024):
    if (url is None) or (str(url).strip() == ""):
        raise ValueError("download_file got empty url")
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {url} -> {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


# =========================================================
# 2) VJEPA2: 权重加载 + transform（纯 PyTorch）
# =========================================================
def load_pretrained_vjepa2_pt_weights(model, pretrained_weights_path: str):
    """
    兼容你贴的 VJEPA2 demo 权重格式：
      ckpt["encoder"]，并且 key 可能带 module./backbone. 前缀
    """
    ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
    if "encoder" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'encoder'. Keys={list(ckpt.keys())[:20]}")
    sd = ckpt["encoder"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[VJEPA2] Loaded encoder weights from {pretrained_weights_path} with msg: {msg}")


def build_vjepa2_video_transform(img_size: int):
    """
    注意：这是给“输入为 T x C x H x W 的 torch video tensor”用的 transform，
    最终输出为 [C, T, H, W]（ClipToTensor 会做维度整理），并 Normalize。
    """
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


# =========================================================
# 3) 你的对齐模块：共同维度 + 原型 + Gram + PCA 子空间
# =========================================================
class SoftKMeans(nn.Module):
    def __init__(self, num_prototypes=256, iters=5, temp=0.5, eps=1e-6):
        super().__init__()
        self.K = num_prototypes
        self.iters = iters
        self.temp = temp
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        x = x.float()
        B, N, D = x.shape

        idx = torch.randint(0, N, (B, self.K), device=x.device)
        c = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))  # [B,K,D]

        for _ in range(self.iters):
            x2 = (x ** 2).sum(dim=-1, keepdim=True)      # [B,N,1]
            c2 = (c ** 2).sum(dim=-1).unsqueeze(1)       # [B,1,K]
            xc = torch.bmm(x, c.transpose(1, 2))         # [B,N,K]
            dist2 = x2 + c2 - 2 * xc                     # [B,N,K]

            w = torch.softmax(-dist2 / max(self.temp, self.eps), dim=-1)  # [B,N,K]
            denom = w.sum(dim=1).unsqueeze(-1) + self.eps                 # [B,K,1]
            c = torch.bmm(w.transpose(1, 2), x) / denom                   # [B,K,D]

        return c


def gram_matrix(tokens, normalize_tokens=True, eps=1e-6):
    if normalize_tokens:
        tokens = F.normalize(tokens, dim=-1, eps=eps)
    return torch.bmm(tokens, tokens.transpose(1, 2))


def off_diagonal(x: torch.Tensor):
    # x: [D,D]
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_pooled_loss(s_tok: torch.Tensor,
                       t_tok: torch.Tensor,
                       sim_w=25.0, var_w=25.0, cov_w=1.0,
                       eps=1e-4):
    """
    s_tok, t_tok: [B, N, D]
    pooled -> [B, D]
    teacher 默认应 detach（在调用处做）
    """
    x = s_tok.mean(dim=1)  # [B,D]
    y = t_tok.mean(dim=1)  # [B,D]

    # invariance
    sim = F.mse_loss(x, y)

    # variance (anti-collapse)
    def var_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    var = var_term(x) + var_term(y)

    # covariance (decorrelation)
    def cov_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        B, D = z.shape
        cov = (z.T @ z) / (B - 1 + 1e-6)   # [D,D]
        return (off_diagonal(cov) ** 2).sum() / D

    cov = cov_term(x) + cov_term(y)

    return sim_w * sim + var_w * var + cov_w * cov, {"vic_sim": sim, "vic_var": var, "vic_cov": cov}

def get_orig_module(module):
    if hasattr(module, 'module'):
        module = module.module
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module

class VJepaAlignerV3(nn.Module):
    def __init__(
        self,
        student_dim,
        teacher_dim,
        student_grid,
        common_dim=512,
        num_prototypes=256,
        kmeans_iters=5,
        kmeans_temp=0.2,
        gram_weight=2,
    ):
        super().__init__()
        self.student_grid = student_grid
        self.common_dim = common_dim
        self.num_prototypes = num_prototypes

        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim),
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim),
        )

        # Assuming SoftKMeans is available in your scope
        self.pool = SoftKMeans(num_prototypes=num_prototypes, iters=kmeans_iters, temp=kmeans_temp)
        self.gram_weight = gram_weight

    def forward(self, student_q, teacher_feats, teacher_grid_shape):
        """
        student_q:    [B, Ns(=t*h*w), Ds]
        teacher_feats:[B, Nt(+1?), Dt]
        teacher_grid_shape: (tt, ht, wt)
        """
        ts, hs, ws = self.student_grid
        tt, ht, wt = teacher_grid_shape
        expected_tokens = tt * ht * wt

        s = self.student_proj(student_q)       # [B,Ns,Dc]
        t = self.teacher_proj(teacher_feats)   # [B,Nt,Dc]

        # student -> grid
        s_3d = rearrange(s, "b (t h w) c -> b c t h w", t=ts, h=hs, w=ws)

        # teacher drop CLS if needed
        if t.shape[1] == expected_tokens + 1:
            t = t[:, 1:, :]
        if t.shape[1] != expected_tokens:
            # Handle potential mismatch gracefully or raise error
            # For robustness, we check shape, if fails, we might slice or error out
            if t.shape[1] > expected_tokens:
                 t = t[:, :expected_tokens, :]
            else:
                raise AssertionError(
                    f"Token mismatch: teacher={t.shape[1]}, expected={expected_tokens} from grid {teacher_grid_shape}"
                )

        t_3d = rearrange(t, "b (t h w) c -> b c t h w", t=tt, h=ht, w=wt)

        # align teacher grid -> student grid
        t_aligned = F.interpolate(t_3d, size=(ts, hs, ws), mode="trilinear", align_corners=False)

        s_tok = rearrange(s_3d, "b c t h w -> b (t h w) c")
        t_tok = rearrange(t_aligned, "b c t h w -> b (t h w) c")

        s_proto = self.pool(s_tok)
        t_proto = self.pool(t_tok)

        gram_loss = F.mse_loss(s_proto, t_proto)
        loss = self.gram_weight * gram_loss
        return loss, {"gram_loss": gram_loss.detach()}

# -----------------------------------------------------------------------------
# Core Output Layer
# -----------------------------------------------------------------------------

class OutputLayer(nn.Module):
    def __init__(self, hidden_size, temporal_patch_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, temporal_patch_size * patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        # x: [b, n, c]
        x = self.norm_final(x)
        x = self.linear(x)
        return x

# -----------------------------------------------------------------------------
# LARP Tokenizer with VJEPA Support
# -----------------------------------------------------------------------------

@register('larp_tokenizer_sem')
class LARPTokenizer(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        bottleneck_token_num=1024,
        input_size=128,
        frame_num=16,
        temporal_patch_size=4,
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,

        use_encoder_patch_token_type_embed=False,
        use_encoder_latent_query_token_type_embed=False,
        use_decoder_latent_token_type_embed=False,
        use_decoder_patch_query_token_type_embed=False,

        encoder_query_gaussian_init=True,
        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vitl.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        # --- Alignment Config ---
        latent_grid_shape=(4, 16, 16), # MUST match bottleneck_token_num (e.g. 4*16*16=1024)
        align_common_dim=256,
        align_num_prototypes=256,
        align_kmeans_iters=5,
        align_kmeans_temp=0.2,
        align_gram_weight=1.0,
        align_pca_weight=0.2, # kept for compatibility if needed later
        align_pca_rank=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.frame_num = frame_num
        self.bottleneck_token_num = bottleneck_token_num
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.decoder_temporal_patch_size = decoder_temporal_patch_size
        self.decoder_patch_size = decoder_patch_size
        self.decoder_latent_len = bottleneck_token_num

        self.encoder_hidden_size = encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = decoder_hidden_size = int(decoder_hidden_size)
        self.encoder_num_heads = encoder_num_heads = int(encoder_num_heads)
        self.decoder_num_heads = decoder_num_heads = int(decoder_num_heads)

        self.latent_pe_scale_factor = latent_pe_scale_factor
        self.query_init_std = query_init_std

        if temporal_patch_size == 1:
            self.x_embedder = VideoPatchEmbed(input_size, patch_size, in_channels, encoder_hidden_size, bias=True, frame_num=frame_num)
        else:
            assert temporal_patch_size > 1
            self.x_embedder = PatchEmbed3D(input_size, frame_num, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True)
        self.token_h = token_h = self.token_w = token_w = int(self.x_embedder.num_spatial_patches ** 0.5)
        self.token_t = token_t = self.x_embedder.num_temporal_patches
        self.video_token_num = video_token_num = self.x_embedder.num_spatial_patches * token_t
        assert input_size % decoder_patch_size == 0, "input_size must be divisible by decoder_patch_size"
        self.decoder_token_t = decoder_token_t = frame_num // decoder_temporal_patch_size
        decoder_token_h = decoder_token_w = input_size // decoder_patch_size
        recon_num_patches_per_frame = decoder_token_h * decoder_token_w
        self.decoder_token_h = self.decoder_token_w = decoder_token_h
        self.recon_video_token_num = recon_video_token_num = recon_num_patches_per_frame * decoder_token_t
        
        # --- Encoder Position Embeddings ---
        self.learned_encoder_patch_pe = learned_encoder_patch_pe
        if self.learned_encoder_patch_pe:
            self.encoder_h_embed = nn.Parameter(torch.zeros(1, 1, token_h, 1, encoder_hidden_size), requires_grad=True)
            self.encode_w_embed = nn.Parameter(torch.zeros(1, 1, 1, token_w, encoder_hidden_size), requires_grad=True)
            self.encoder_t_embed = nn.Parameter(torch.zeros(1, token_t, 1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe_raw = lambda: (self.encoder_h_embed + self.encode_w_embed + self.encoder_t_embed).reshape(1, video_token_num, encoder_hidden_size)
        else:
            self.register_buffer('encoder_patch_pe', torch.zeros(1, video_token_num, encoder_hidden_size))
            self.get_encoder_patch_pe_raw = lambda: self.encoder_patch_pe
        self.use_encoder_patch_token_type_embed = use_encoder_patch_token_type_embed
        if self.use_encoder_patch_token_type_embed:
            self.encoder_patch_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe = lambda: self.get_encoder_patch_pe_raw() + self.encoder_patch_token_type_embed
        else:
            self.get_encoder_patch_pe = self.get_encoder_patch_pe_raw

        # --- Encoder Latent Query Embeddings ---
        self.learned_encoder_latent_query_embed = learned_encoder_latent_query_embed
        self.encoder_query_gaussian_init = encoder_query_gaussian_init
        if self.learned_encoder_latent_query_embed:
            self.encoder_latent_query_embed = nn.Parameter(torch.zeros(bottleneck_token_num, encoder_hidden_size), requires_grad=True)
        else:
            self.register_buffer('encoder_latent_query_embed', torch.zeros(bottleneck_token_num, encoder_hidden_size))
            assert not encoder_query_gaussian_init, "encoder_query_gaussian_init requires learned_encoder_latent_query_embed to be True"
        self.use_encoder_latent_query_token_type_embed = use_encoder_latent_query_token_type_embed
        if self.use_encoder_latent_query_token_type_embed:
            self.encoder_latent_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0) + self.encoder_latent_query_token_type_embed
        else:
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0)

        # --- Decoder Latent Position Embeddings ---
        self.learned_decoder_latent_pe = learned_decoder_latent_pe
        if self.learned_decoder_latent_pe:
            self.decoder_latent_pe = nn.Parameter(torch.zeros(1, self.decoder_latent_len, decoder_hidden_size), requires_grad=True)
        else: 
            self.register_buffer('decoder_latent_pe', torch.zeros(1, self.decoder_latent_len, decoder_hidden_size))
        self.use_decoder_latent_token_type_embed = use_decoder_latent_token_type_embed
        if self.use_decoder_latent_token_type_embed:
            self.decoder_latent_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe + self.decoder_latent_token_type_embed
        else: 
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe

        # --- Decoder Patch Query Embeddings ---
        self.learned_decoder_patch_query_embed = learned_decoder_patch_query_embed
        if self.learned_decoder_patch_query_embed:
            self.decoder_h_embed = nn.Parameter(torch.zeros(1, 1, decoder_token_h, 1, decoder_hidden_size), requires_grad=True)
            self.decoder_w_embed = nn.Parameter(torch.zeros(1, 1, 1, decoder_token_w, decoder_hidden_size), requires_grad=True)
            self.decoder_t_embed = nn.Parameter(torch.zeros(1, decoder_token_t, 1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed_raw = lambda: (self.decoder_h_embed + self.decoder_w_embed + self.decoder_t_embed).reshape(1, recon_video_token_num, decoder_hidden_size)
        else:
            self.register_buffer('decoder_patch_query_embed', torch.zeros(1, recon_video_token_num, decoder_hidden_size))
            self.get_decoder_patch_query_embed_raw = lambda: self.decoder_patch_query_embed
        self.use_decoder_patch_query_token_type_embed = use_decoder_patch_query_token_type_embed
        if self.use_decoder_patch_query_token_type_embed:
            self.decoder_patch_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed = lambda: self.get_decoder_patch_query_embed_raw() + self.decoder_patch_query_token_type_embed
        else: 
            self.get_decoder_patch_query_embed = self.get_decoder_patch_query_embed_raw


        # --- Models ---
        if encoder_name is None or encoder_name.lower() in ['none', 'no', 'null', '']:
            encoder_name = transformer_name
        if decoder_name is None or decoder_name.lower() in ['none', 'no', 'null', '']:
            decoder_name = transformer_name

        encoder_args = {
            'name': encoder_name,
            'args': {
                'dim': encoder_hidden_size,
                'depth': encoder_depth,
                'n_head': encoder_num_heads,
                'head_dim': encoder_hidden_size // encoder_num_heads,
            }, 
        }

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, 
        }

        self.encoder = models.make(encoder_args)
        self.decoder = models.make(decoder_args)

        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
        bottleneck_args = {'token_nums': self.bottleneck_token_num, 'input_dim': encoder_hidden_size, 'output_dim': decoder_hidden_size}
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']
        self.final_layer = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)

        # Prior model
        prior_model = edict(prior_model)
        self.prior_model = None

        # --- VJEPA / Alignment Initialization ---
        self.use_vjepa_loss = use_vjepa_loss
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self.teacher_model = None

        # Align Config
        self.latent_grid_shape = latent_grid_shape
        if np.prod(latent_grid_shape) != bottleneck_token_num:
            raise ValueError(f"latent_grid_shape {latent_grid_shape} must match bottleneck_token_num {bottleneck_token_num}")
            
        self.align_common_dim = align_common_dim
        self.align_num_prototypes = align_num_prototypes
        self.align_kmeans_iters = align_kmeans_iters
        self.align_kmeans_temp = align_kmeans_temp
        self.align_gram_weight = align_gram_weight
        self.align_pca_weight = align_pca_weight
        self.align_pca_rank = align_pca_rank

        if self.use_vjepa_loss:
            # We treat the bottleneck dimension as the student dimension
            self._init_vjepa2_teacher(student_dim=self.decoder_hidden_size)

        self.initialize_weights()

    def _init_vjepa2_teacher(self, student_dim: int):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return

        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # NOTE: requires vit_large_rope and load_pretrained_vjepa2_pt_weights available in scope/imports
        self.teacher_model = vit_large_rope(
            img_size=(self.vjepa2_img_size, self.vjepa2_img_size),
            num_frames=self.vjepa2_num_frames,
        )
        load_pretrained_vjepa2_pt_weights(self.teacher_model, self.vjepa2_encoder_ckpt)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        if self.vjepa2_use_bf16:
            self.teacher_model = self.teacher_model.to(dtype=torch.bfloat16)

        teacher_dim = getattr(self.teacher_model, "embed_dim", None)
        if teacher_dim is None:
            raise AttributeError("teacher_model has no attribute embed_dim; please check VJEPA2 model class.")
        print(f"[VJEPA2] Teacher loaded. embed_dim={teacher_dim}, img_size={self.vjepa2_img_size}, num_frames={self.vjepa2_num_frames}")

        # teacher transform (PyTorch)
        self.teacher_transform = build_vjepa2_video_transform(img_size=self.vjepa2_img_size)
        
        # aligner
        self.aligner = VJepaAlignerV3(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            student_grid=self.latent_grid_shape,
            common_dim=self.align_common_dim,
            num_prototypes=self.align_num_prototypes,
            kmeans_iters=self.align_kmeans_iters,
            kmeans_temp=self.align_kmeans_temp,
            gram_weight=self.align_gram_weight,
        )

    def _sample_or_pad_frames(self, x: torch.Tensor, out_T: int):
        """
        x: [B, 3, T, H, W]
        return: [B, 3, out_T, H, W]
        """
        B, C, T, H, W = x.shape
        if T == out_T:
            return x
        if T > out_T:
            if self.vjepa2_sample_strategy == "uniform":
                idx = torch.linspace(0, T - 1, out_T, device=x.device).round().long()
                return x[:, :, idx, :, :]
            # default: take first out_T
            return x[:, :, :out_T, :, :]
        # T < out_T
        if self.vjepa2_sample_strategy == "uniform" and T > 1:
            idx = torch.linspace(0, T - 1, out_T, device=x.device).round().long()
            return x[:, :, idx, :, :]
        # default: repeat last frame
        pad = x[:, :, -1:, :, :].expand(B, C, out_T - T, H, W)
        return torch.cat([x, pad], dim=2)

    def _preprocess_for_vjepa2(self, x: torch.Tensor):
        """
        x: [B, 3, T, H, W]
        """
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.float()

        # NOTE: You might want to sample frames here if input T != teacher T
        # x = self._sample_or_pad_frames(x, self.vjepa2_num_frames) 

        outs = []
        for b in range(x.shape[0]):
            vid_tchw = x[b].permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]
            out_cthw = self.teacher_transform(vid_tchw)       # [C,T,Ht,Wt]
            outs.append(out_cthw)
        x_teacher = torch.stack(outs, dim=0)  # [B,C,T,Ht,Wt]

        if self.teacher_model is not None:
            target_dtype = next(self.teacher_model.parameters()).dtype
            x_teacher = x_teacher.to(dtype=target_dtype)

        return x_teacher

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        token_h, token_w = self.token_h, self.token_w

        # Initialize encoder patch PE
        if self.learned_encoder_patch_pe:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.token_t))
            self.encoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.encoder_h_embed))
            self.encode_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.encode_w_embed))
            self.encoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.encoder_t_embed))
        else:
            encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t)
            self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))
        if self.use_encoder_patch_token_type_embed:
            encoder_patch_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_patch_token_type_embed.data.copy_(encoder_patch_token_type_embed)

        # Initialize encoder latent query embed
        if self.learned_encoder_latent_query_embed:
            if self.encoder_query_gaussian_init:
                query_embed = torch.randn(self.bottleneck_token_num, self.encoder_hidden_size) * self.query_init_std
            else:
                query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num))
                query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        else:
            query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num), self.latent_pe_scale_factor)
            query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        self.encoder_latent_query_embed.data.copy_(query_embed)
        if self.use_encoder_latent_query_token_type_embed:
            encoder_latent_query_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_latent_query_token_type_embed.data.copy_(encoder_latent_query_token_type_embed)

        # initialize decoder latent PE
        if self.learned_decoder_latent_pe:
            decoder_token_embed = torch.randn(1, self.decoder_latent_len, self.decoder_hidden_size) * .02
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        else:
            decoder_token_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_latent_len), self.latent_pe_scale_factor)
            decoder_token_embed = torch.from_numpy(decoder_token_embed).float().reshape(1, self.decoder_latent_len, self.decoder_hidden_size)
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        if self.use_decoder_latent_token_type_embed:
            decoder_latent_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_latent_token_type_embed.data.copy_(decoder_latent_token_type_embed)

        # initialize decoder patch query PE
        if self.learned_decoder_patch_query_embed:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_t))
            self.decoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.decoder_h_embed))
            self.decoder_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.decoder_w_embed))
            self.decoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.decoder_t_embed))
        else:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.decoder_token_h, self.decoder_token_t)
            self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        if self.use_decoder_patch_query_token_type_embed:
            decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_last_layer(self):
        return self.final_layer.linear.weight

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decoder_parameters(self):
        decoder_params = itertools.chain(
            self.decoder.parameters(),
            self.final_layer.parameters()
        )
        if self.learned_decoder_patch_query_embed:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_h_embed, self.decoder_w_embed, self.decoder_t_embed]
            )
        if self.learned_decoder_latent_pe:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_latent_pe]
            )
        return decoder_params

    def decoder_requires_grad_(self, requires_grad):
        for param in self.decoder_parameters():
            param.requires_grad_(requires_grad)

    def others_parameters(self):
        decoder_params_set = set(self.decoder_parameters())
        return (p for p in self.parameters() if p not in decoder_params_set)

    def others_requires_grad_(self, requires_grad):
        for param in self.others_parameters():
            param.requires_grad_(requires_grad)

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True, version='sd'):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f"checkpoint must be a dict or a path to a checkpoint"

        kwargs = ckpt["model"]["args"]
        model = cls(**kwargs)
        if load_state_dict:
            if version == 'sd':
                sd = ckpt["model"]["sd"]
            elif version.startswith('ema'):
                assert '_' in version, "ema version must be in the format 'ema_{alpha}'"
                alpha = float(version.split('_')[1])
                sd = ckpt["model"]['ema_sd'][alpha]
            else:
                raise ValueError(f"Unknown version: {version}")
            model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x):
        x = self.x_embedder(x) + self.get_encoder_patch_pe() # (b, n, d)
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) # (b, n, d)
        z = self.encoder(x, q_emb)
        bottleneck_out = self.bottleneck(z)
        z = bottleneck_out.pop('output')
        return {'encoded': z, **bottleneck_out}

    def encode_eval(self, x):
        x_tokens = self.x_embedder(x)
        num_x_tokens = x_tokens.size(1)
        x = x_tokens + self.get_encoder_patch_pe()[:, :num_x_tokens, :] 
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) 
        z = self.encoder(x, q_emb)
        bottleneck_out = self.bottleneck(z)
        z = bottleneck_out.pop('output')
        return {'encoded': z, **bottleneck_out, 'num_x_tokens': num_x_tokens}

    def unpatchify(self, x):
        """
        x: (b, n, t_patch_size * s_patch_size**2 * c)
        videos: (b, c, t, h, w)
        """
        c = self.out_channels
        pt = self.temporal_patch_size
        p = self.patch_size
        h = w = self.token_h
        t = x.size(1) // (h * w)
        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        b = z.size(0)
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(b, -1, -1)
        x = self.decoder(z, decoder_pos_embed)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

    def decode_eval(self, z, num_x_tokens=None):
        b = z.size(0)
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(b, -1, -1)
        if num_x_tokens is not None:
            decoder_pos_embed = decoder_pos_embed[:, :num_x_tokens, :]
        x = self.decoder(z, decoder_pos_embed)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

    def decode_from_bottleneck(self, bottleneck_rep):
        z = self.bottleneck.decode(bottleneck_rep) 
        return self.decode(z)

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        B = data.size(0)
        encode_output = self.encode(data)
        
        # 'encoded' here is the latent representation z [B, N, D]
        # This is what we will align with the teacher
        latents = encode_output['encoded'] 

        pred_frames = self.decode(latents).contiguous() # [b, c, t, h, w]
        return_dict = {'pred_frames': pred_frames, **encode_output}

        # --- VJEPA Alignment Loss Calculation ---
        if self.training and self.use_vjepa_loss and (self.teacher_model is not None):
            # Move teacher if needed
            if next(self.teacher_model.parameters()).device != data.device:
                self.teacher_model.to(data.device)

            with torch.no_grad():
                # Preprocess data for Teacher (Resize, Norm, etc.)
                t_input = self._preprocess_for_vjepa2(data)     # [B,3,T',Ht,Wt]
                t_feats = self.teacher_model(t_input)           # [B,Nt,Dt]
                t_feats = t_feats.float()

            # Calculate teacher grid shape based on config
            tt = self.vjepa2_num_frames // self.vjepa2_tubelet_size
            ht = self.vjepa2_img_size // self.vjepa2_patch_size
            wt = self.vjepa2_img_size // self.vjepa2_patch_size

            # The student input to aligner is the latent representation (usually float)
            # Some bottlenecks return quantized indices, here 'encoded' is usually the vector
            align_loss, align_dict = self.aligner(latents.float(), t_feats, (tt, ht, wt))
            
            return_dict["align_loss"] = 0.5 * align_loss
            # return_dict.update({f"align_{k}": v for k, v in align_dict.items()})

        return return_dict