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
from .transformer import DEC
from .embed import (PatchEmbed3D, VideoPatchEmbed,
                    get_1d_sincos_pos_embed_from_grid, get_3d_sincos_pos_embed)

from vjepa2.src.models.vision_transformer import vit_huge_rope
import vjepa2.src.datasets.utils.video.transforms as video_transforms
import vjepa2.src.datasets.utils.video.volume_transforms as volume_transforms
from models.model_new.quantizer.fsq import FSQ,VectorQuantizer
def get_orig_module(module):
    if hasattr(module, 'module'):
        module = module.module
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module

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


class LightweightSemanticInjector(nn.Module):
    def __init__(self, dim, reduction_ratio=128, kernel_size=3):
        super().__init__()
        self.dim = dim
        hidden_dim = dim // reduction_ratio
        padding = kernel_size // 2
        
        # 1. 归一化
        self.norm_shallow = nn.GroupNorm(32, dim)

        # 2. 参数生成网络
        # A. 降维 (Linear)
        self.proj_down = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )
        
        # B. 空间混合 (Depthwise 3D Conv)
        # groups=hidden_dim 是省参数的关键
        self.spatial_mix = nn.Conv3d(
            hidden_dim, hidden_dim, 
            kernel_size=kernel_size, padding=padding, 
            groups=hidden_dim 
        )
        
        # C. 升维 (Linear)
        self.proj_up = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )
        
        # 零初始化
        nn.init.constant_(self.proj_up[-1].weight, 0)
        nn.init.constant_(self.proj_up[-1].bias, 0)

    def forward(self, x_shallow, x_deep, shape_info):
        B, N, D = x_shallow.shape
        T, H, W = shape_info
        
        # --- 1. 生成调制参数 (Parameter Generation) ---
        # Input: Deep features [B, N, D]
        
        # Step A: 降维 [B, N, D/4]
        h = self.proj_down(x_deep)
        
        # Step B: 空间混合
        # 变换为 [B, D/4, T, H, W] 以进行 3D 卷积
        h_3d = h.transpose(1, 2).reshape(B, -1, T, H, W)
        h_3d = self.spatial_mix(h_3d)
        # 变回 [B, N, D/4]
        h = h_3d.flatten(2).transpose(1, 2)
        
        # Step C: 升维 [B, N, 2D]
        style = self.proj_up(h)
        
        # --- 2. 注入调制 (Injection) ---
        # 将 style 变回 [B, 2D, T, H, W]
        style_3d = style.transpose(1, 2).reshape(B, 2*D, T, H, W)
        scale, shift = style_3d.chunk(2, dim=1)
        scale = scale + 1.0
        
        # 处理 shallow 特征
        shallow_3d = x_shallow.transpose(1, 2).reshape(B, D, T, H, W)
        shallow_norm = self.norm_shallow(shallow_3d)
        
        # AdaIN / SPADE logic
        out_3d = shallow_norm * scale + shift
        
        # Residual Connection
        out_3d = out_3d + shallow_3d
        
        return out_3d.flatten(2).transpose(1, 2)

class SemanticPyramidFusion(nn.Module):
    def __init__(self, dim, grid_size):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        
        # 使用轻量级模块
        self.injector_l24 = LightweightSemanticInjector(dim)
        self.injector_l16 = LightweightSemanticInjector(dim)
        self.injector_l8 = LightweightSemanticInjector(dim)
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, feats_list):
        assert len(feats_list) == 4
        f_l8, f_l16, f_l24, f_l32 = feats_list
        
        # Top-Down Pathway
        feat_flow = self.injector_l24(f_l24, f_l32, self.grid_size)
        feat_flow = self.injector_l16(f_l16, feat_flow, self.grid_size)
        feat_flow = self.injector_l8(f_l8, feat_flow, self.grid_size)
        
        return self.final_proj(feat_flow)



class GatedLinearLayerFusion(nn.Module):
    """
    Per-layer gated fusion:
        gate_l = sigmoid(MLP(LN(f_l)))        # [B,N,1] token-wise gate
        proj_l = Linear(LN(f_l))              # [B,N,D]
        fused  = sum_l gate_l * proj_l        # [B,N,D]
        out    = LN(fused)

    Input:  feats_list: list of L tensors, each [B,N,D]
    Output: fused: [B,N,D]
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        gate_hidden_ratio: float = 0.25,
        use_pre_ln: bool = True,
        use_post_ln: bool = True,
        per_layer_proj: bool = True,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.use_pre_ln = bool(use_pre_ln)
        self.use_post_ln = bool(use_post_ln)
        self.per_layer_proj = bool(per_layer_proj)

        self.pre_ln = nn.LayerNorm(self.dim) if self.use_pre_ln else nn.Identity()
        self.post_ln = nn.LayerNorm(self.dim) if self.use_post_ln else nn.Identity()

        hidden = max(1, int(self.dim * gate_hidden_ratio))

        # One gate MLP per layer (token-wise -> scalar gate)
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(self.num_layers)
        ])

        # Optional per-layer projection D->D
        if self.per_layer_proj:
            self.projs = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)])
        else:
            self.projs = nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])

        # Initialize last gate layer bias to control initial openness
        for mlp in self.gate_mlps:
            last = mlp[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                nn.init.constant_(last.bias, gate_bias_init)

    def forward(self, feats_list):
        assert isinstance(feats_list, (list, tuple)), "feats_list must be list/tuple"
        assert len(feats_list) == self.num_layers, f"Expected {self.num_layers} layers, got {len(feats_list)}"

        fused = None
        for i, f in enumerate(feats_list):
            # f: [B,N,D]
            x = self.pre_ln(f)
            gate = torch.sigmoid(self.gate_mlps[i](x))   # [B,N,1]
            proj = self.projs[i](x)                      # [B,N,D]
            contrib = gate * proj                        # broadcast over D

            fused = contrib if fused is None else (fused + contrib)

        fused = self.post_ln(fused)
        return fused













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
    #short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(256, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

@register('larp_tokenizer_vfm')
class LARPTokenizer(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        bottleneck_token_num=1024,
        input_size=256,
        frame_num=16,
        temporal_patch_size=2,
        patch_size=16,
        decoder_temporal_patch_size=2,
        decoder_patch_size=16,
        in_channels=3,
        bottleneck_type='auto',
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
        train_type='simple',
        fusionmode='gate',  # 'gate' | 'sem' | 'concat'
        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,
        use_pe='yes',
        use_encoder_patch_token_type_embed=False,
        use_encoder_latent_query_token_type_embed=False,
        use_decoder_latent_token_type_embed=False,
        use_decoder_patch_query_token_type_embed=False,

        encoder_query_gaussian_init=True,


        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

    ):
        super().__init__()
        self.use_pe = use_pe
        self.fusionmode = fusionmode
        self.train_type = train_type
        self.bottleneck_type = bottleneck_type
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


        self.token_h = token_h = self.token_w = token_w = 16
        self.token_t = token_t = 8
        self.video_token_num = video_token_num =2048
        assert input_size % decoder_patch_size == 0, "input_size must be divisible by decoder_patch_size"
        self.decoder_token_t = decoder_token_t = frame_num // decoder_temporal_patch_size
        print(f"DEBUG: frame_num={frame_num}, decoder_temporal_patch_size={decoder_temporal_patch_size}, decoder_token_t={decoder_token_t}")
        decoder_token_h = decoder_token_w = input_size // decoder_patch_size
        recon_num_patches_per_frame = decoder_token_h * decoder_token_w
        self.decoder_token_h = self.decoder_token_w = decoder_token_h
        print(f"DEBUG: decoder_token_h={decoder_token_h}, decoder_token_w={decoder_token_w}, recon_num_patches_per_frame={recon_num_patches_per_frame}")
        print(decoder_token_t)
        self.recon_video_token_num = recon_video_token_num = recon_num_patches_per_frame * decoder_token_t
        # encoder patch PE
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

        # encoder latent query embed        learned_encoder_latent_query_embed: true
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

        # decoder latent PE
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

        # decoder latent PE

        self.register_buffer('imagedec_latent_pe', torch.zeros(1, 2048, 1024))
        self.imagedec_latent_token_type_embed = nn.Parameter(torch.zeros(1, 1, 1024), requires_grad=True)
        self.get_imagedec_latent_pe = lambda: self.imagedec_latent_pe + self.imagedec_latent_token_type_embed






        # decoder patch query embed
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


        # Build encoder, decoder, and bottleneck
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
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        self.decoder_args_dec_to_image = DEC(1024,24,16,64)

        self.encoder = models.make(encoder_args)
        self.decoder = models.make(decoder_args)

        self.jepa_to_encoder= nn.Linear(1280, encoder_hidden_size)  # 适配 VJEPA2 teacher 特征维度到 encoder 输入维度
        self.dec_to_decimage= nn.Linear(decoder_hidden_size,1024)  # 适配 decoder 输出到 DEC 输入维度


        if self.bottleneck_type == 'vq':
            self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
            bottleneck_args = {'token_nums': self.bottleneck_token_num, 'input_dim': encoder_hidden_size, 'output_dim': decoder_hidden_size}
            self.bottleneck = models.make(bottleneck, args=bottleneck_args)
            self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']
        elif self.bottleneck_type == 'sq':
            self.sq_in_linear = nn.Linear(self.encoder_hidden_size, 24)
            self.sq_out_linear = nn.Linear(24, self.decoder_hidden_size)
            self.bottleneck = VectorQuantizer(n_embed=196_560, embed_dim=24, l2_norm=True, beta=0.25, input_format='blc')



        self.final_layer = OutputLayer(1024, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)
        # Build prior model
        prior_model = edict(prior_model)
        # if prior_model.get('name', '').lower() in ['none', 'no', 'null', '']:
        self.prior_model = None
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher()
        self.aligner= nn.Linear(decoder_hidden_size, 1280) 
        self.teacher_layer_norms = nn.ModuleList([
            nn.LayerNorm(1280, eps=1e-6) for _ in range(4)
        ])



        if self.fusionmode == 'gate':
            self.fusion_proj = GatedLinearLayerFusion(
                dim=1280,  # default teacher dim; will adjust to teacher.embed_dim at init
                num_layers=4,
            )
        elif self.fusionmode == 'sem':
             self.fusion_proj =SemanticPyramidFusion(
                dim=1280,
                grid_size=self.vfm_grid
            )
        else:
            self.fusion_proj = nn.Sequential(
            #nn.LayerNorm(1280*4),      # 关键：先归一化，平衡不同层的数值范围
                nn.Linear(1280*4, 1280), # 投影回 1280
                nn.GELU()                      # 可选：增加非线性，通常有助于特征混合
            )



        self.initialize_weights()



    def _init_vjepa2_teacher(self):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return
        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
        self.teacher_model = vit_huge_rope(
            img_size=(self.vjepa2_img_size, self.vjepa2_img_size),
            num_frames=self.vjepa2_num_frames,
            out_layers=[8, 16, 24, 31],
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
        self.teacher_transform = build_vjepa2_video_transform(img_size=self.vjepa2_img_size)
        self.prior_model = None 

    def _preprocess_for_vjepa2(self, x: torch.Tensor):
        """
        x: [B, 3, T, H, W], 期望是 0..1 的浮点
        输出：x_teacher [B, 3, T', Ht, Wt]，已做 resize/crop + imagenet normalize
        """
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.float()#B,3,T,H,W
        outs = []
        for b in range(x.shape[0]):
            vid_tchw = x[b].permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]
            out_cthw = self.teacher_transform(vid_tchw)       # [C,T,Ht,Wt]
            outs.append(out_cthw)
        x_teacher = torch.stack(outs, dim=0)  # [B,C,T,Ht,Wt]

        # dtype 对齐到 teacher
        if self.teacher_model is not None:
            target_dtype = next(self.teacher_model.parameters()).dtype
            x_teacher = x_teacher.to(dtype=target_dtype)

        return x_teacher

    @torch.no_grad()
    def _extract_vfm_features(self, x: torch.Tensor):
        """
        x: raw video [B, 3, T, H, W]
        Returns: [B, num_vfm_tokens, D_teacher]
        """
        assert self.teacher_model is not None, "Teacher model is required for encoding."
        if next(self.teacher_model.parameters()).device != x.device:
            self.teacher_model.to(x.device)
        t_input = self._preprocess_for_vjepa2(x)
        # 1. 获取特征列表
        # 假设 teacher 返回 [feat_l8, feat_l16, feat_l24, feat_l32]
        t_out = self.teacher_model(t_input)
        if self.fusionmode == 'gate':
            fused_feats = self.fusion_proj(t_out).float()  # [B, N, 1280]
        elif self.fusionmode == 'sem':
            if not isinstance(t_out, (list, tuple)):
            # 兼容性处理：如果只返回了一个 Tensor，包一层
                t_out = [t_out]

        # 确保只取需要的 Tensor（有些实现可能返回 tuple）
            feats_list = []
            for f in t_out:
                if isinstance(f, (tuple, list)):
                    f = f[0]
                feats_list.append(f) # List of [B, N, D]

            fused_feats = self.fusion_proj(feats_list)
        else:
            if not isinstance(t_out, (list, tuple)):
                # 兼容性处理：如果只返回了一个 Tensor，包一层
                t_out = [t_out]

            # 确保只取需要的 Tensor（有些实现可能返回 tuple）
            feats_list = []
            for f in t_out:
                if isinstance(f, (tuple, list)):
                    f = f[0]
                feats_list.append(f) # List of [B, N, D]
            normed_feats = []
            for i, feat in enumerate(feats_list):
                # feat: [B, N, 1280]
                normed_feats.append(self.teacher_layer_norms[i](feat))
            # 2. 拼接 (Concatenation)
            # [B, N, D] * 4 -> [B, N, 4*D]
            # 必须转 float 以便进行 LayerNorm 和 Linear 计算（避免 bf16 溢出或精度问题）
            cat_feats = torch.cat(normed_feats, dim=-1).float() 
            # 3. 线性映射融合 (Linear Projection)
            # [B, N, 4*D] -> [B, N, 1280]
            fused_feats = self.fusion_proj(cat_feats)
        return fused_feats






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
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t)
            self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))
        if self.use_encoder_patch_token_type_embed:
            encoder_patch_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_patch_token_type_embed.data.copy_(encoder_patch_token_type_embed)

        # Initialize encoder latent query embed
        if self.learned_encoder_latent_query_embed:
            if self.encoder_query_gaussian_init:
                # from timm vision_transformer.py
                # https://github.com/huggingface/pytorch-image-models/blob/70ccf00c95a2d78a166cca24ef6adbca46f47c2a/timm/models/vision_transformer.py#L495
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




        imagedec_token_embed = get_3d_sincos_pos_embed(1024, 16, 8)
        imagedec_token_embed = torch.from_numpy(imagedec_token_embed).float().reshape(1, 2048, 1024)
        self.imagedec_latent_pe.data.copy_(imagedec_token_embed)
        imagedec_latent_token_type_embed = torch.randn(1, 1, 1024) * .02
        self.imagedec_latent_token_type_embed.data.copy_(imagedec_latent_token_type_embed)




        # initialize decoder patch query PE
        if self.learned_decoder_patch_query_embed:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_t))
            self.decoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.decoder_h_embed))
            self.decoder_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.decoder_w_embed))
            self.decoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.decoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.decoder_token_h, self.decoder_token_t)
            self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        if self.use_decoder_patch_query_token_type_embed:
            decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)



        # Zero-out output layers:
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



    def encode(self, x):
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]
        vfm_feats_encode = self.jepa_to_encoder(vfm_feats)          # [B, 2048, encoder_hidden_size]
        x = vfm_feats_encode + self.get_encoder_patch_pe() # (b, n, d)
        #print("x_embedder output shape:", x.shape)
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) # (b, n, d)
        z = self.encoder(x, q_emb)
        if self.bottleneck_type == 'vq':
            bottleneck_out = self.bottleneck(z)
            z= bottleneck_out.pop('output')
        elif self.bottleneck_type == 'sq':
            z = self.sq_in_linear(z)
            bottleneck_out = self.bottleneck(z)
            z= bottleneck_out.pop('output')
            z = self.sq_out_linear(z)
        return {'encoded': z,'vfm_feats':vfm_feats,**bottleneck_out}



    def unpatchify(self, x):
        c = 3
        pt = 2
        p = 16
        h = w = 16
        t = x.size(1) // (h * w)
        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        # z: (b, n, d)
        b = z.size(0)
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(b, -1, -1)
        #print('fuck')
        #print(decoder_pos_embed.shape)
        x = self.decoder(z, decoder_pos_embed)
        dec_vfm=x
        #print(f"DEBUG: decoder output x shape: {x.shape}, decoder_pos_embed shape: {decoder_pos_embed.shape}")
        imagedec_token_embed = self.get_imagedec_latent_pe() 
        imagedec_token_embed = imagedec_token_embed.expand(b, -1, -1)
        x= self.dec_to_decimage(x) 
        #print(f"DEBUG: x shape: {x.shape}, imagedec_token_embed shape: {imagedec_token_embed.shape}") 
        if self.use_pe=='yes':
            x = x + imagedec_token_embed
        # = x + imagedec_token_embed
        #x= self.dec_to_decimage(x)
        x = self.decoder_args_dec_to_image(x)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x, dec_vfm

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        B = data.size(0)
        encode_output = self.encode(data)
        vfm_feats = encode_output['vfm_feats']
        #pred_frames,dec_vfm = self.decode(encode_output['encoded']).contiguous() # [b, c, t, h, w]
        #print(f"DEBUG: encode_output['encoded'] shape: {encode_output['encoded'].shape}")
        pred_frames, dec_vfm = self.decode(encode_output['encoded'])
        pred_frames = pred_frames.contiguous()
        return_dict = {'pred_frames': pred_frames, **encode_output}
        align_student = self.aligner(dec_vfm.float()) # [B, 2048, 1280]
        target = vfm_feats.float().detach()
        student_flat = align_student.reshape(-1, 1280)
        target_flat = target.reshape(-1, 1280)
        cos_loss = 1.0 - F.cosine_similarity(student_flat, target_flat, dim=-1).mean()
        mse_loss = F.mse_loss(align_student, target)
        # F. 组合 Loss
        # 推荐权重：Cos为主 (1.0)，MSE为辅 (0.1 或 0.2)
        return_dict["align_loss"] = 1.0 * cos_loss + 0.1 * mse_loss



        #return_dict = {'pred_frames': pred_frames, **encode_output}   imageddec

        return return_dict













@register('larp_tokenizer_vfm_noquant')
class LARPTokenizer(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        bottleneck_token_num=1024,
        input_size=256,
        frame_num=16,
        temporal_patch_size=2,
        patch_size=16,
        decoder_temporal_patch_size=2,
        decoder_patch_size=16,
        in_channels=3,
        bottleneck_type='auto',
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
        train_type='simple',
        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,
        use_pe='yes',
        use_encoder_patch_token_type_embed=False,
        use_encoder_latent_query_token_type_embed=False,
        use_decoder_latent_token_type_embed=False,
        use_decoder_patch_query_token_type_embed=False,

        encoder_query_gaussian_init=True,


        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

    ):
        super().__init__()
        self.use_pe = use_pe
        self.train_type = train_type
        self.bottleneck_type = bottleneck_type
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


        self.token_h = token_h = self.token_w = token_w = 16
        self.token_t = token_t = 8
        self.video_token_num = video_token_num =2048
        assert input_size % decoder_patch_size == 0, "input_size must be divisible by decoder_patch_size"
        self.decoder_token_t = decoder_token_t = frame_num // decoder_temporal_patch_size
        print(f"DEBUG: frame_num={frame_num}, decoder_temporal_patch_size={decoder_temporal_patch_size}, decoder_token_t={decoder_token_t}")
        decoder_token_h = decoder_token_w = input_size // decoder_patch_size
        recon_num_patches_per_frame = decoder_token_h * decoder_token_w
        self.decoder_token_h = self.decoder_token_w = decoder_token_h
        print(f"DEBUG: decoder_token_h={decoder_token_h}, decoder_token_w={decoder_token_w}, recon_num_patches_per_frame={recon_num_patches_per_frame}")
        print(decoder_token_t)
        self.recon_video_token_num = recon_video_token_num = recon_num_patches_per_frame * decoder_token_t
        # encoder patch PE
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

        # encoder latent query embed        learned_encoder_latent_query_embed: true
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

        # decoder latent PE
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

        # decoder latent PE

        self.register_buffer('imagedec_latent_pe', torch.zeros(1, 2048, 768))
        self.imagedec_latent_token_type_embed = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=True)
        self.get_imagedec_latent_pe = lambda: self.imagedec_latent_pe + self.imagedec_latent_token_type_embed






        # decoder patch query embed
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


        # Build encoder, decoder, and bottleneck
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
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        self.decoder_args_dec_to_image = DEC(768,16,12,64)

        self.encoder = models.make(encoder_args)
        self.decoder = models.make(decoder_args)

        self.jepa_to_encoder= nn.Linear(1280, encoder_hidden_size)  # 适配 VJEPA2 teacher 特征维度到 encoder 输入维度
        self.dec_to_decimage= nn.Linear(1280,768)  # 适配 decoder 输出到 DEC 输入维度


        if self.bottleneck_type == 'vq':
            self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
            bottleneck_args = {'token_nums': self.bottleneck_token_num, 'input_dim': encoder_hidden_size, 'output_dim': decoder_hidden_size}
            self.bottleneck = models.make(bottleneck, args=bottleneck_args)
            self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']
        elif self.bottleneck_type == 'sq':
            self.sq_in_linear = nn.Linear(self.encoder_hidden_size, 24)
            self.sq_out_linear = nn.Linear(24, self.decoder_hidden_size)
            self.bottleneck = VectorQuantizer(n_embed=196_560, embed_dim=24, l2_norm=True, beta=0.25, input_format='blc')



        self.final_layer = OutputLayer(768, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)
        # Build prior model
        prior_model = edict(prior_model)
        # if prior_model.get('name', '').lower() in ['none', 'no', 'null', '']:    1024
        self.prior_model = None
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher()
        self.aligner= nn.Linear(decoder_hidden_size, 1280) 
        self.teacher_layer_norms = nn.ModuleList([
            nn.LayerNorm(1280, eps=1e-6) for _ in range(4)
        ])
        self.fusion_proj = nn.Sequential(
            #nn.LayerNorm(1280*4),      # 关键：先归一化，平衡不同层的数值范围
            nn.Linear(1280*4, 1280), # 投影回 1280
            nn.GELU()                      # 可选：增加非线性，通常有助于特征混合
        )
        self.initialize_weights()



    def _init_vjepa2_teacher(self):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return
        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
        self.teacher_model = vit_huge_rope(
            img_size=(self.vjepa2_img_size, self.vjepa2_img_size),
            num_frames=self.vjepa2_num_frames,
            out_layers=[8, 16, 24, 31],
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
        self.teacher_transform = build_vjepa2_video_transform(img_size=self.vjepa2_img_size)
        self.prior_model = None 

    def _preprocess_for_vjepa2(self, x: torch.Tensor):
        """
        x: [B, 3, T, H, W], 期望是 0..1 的浮点
        输出：x_teacher [B, 3, T', Ht, Wt]，已做 resize/crop + imagenet normalize
        """
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.float()#B,3,T,H,W
        outs = []
        for b in range(x.shape[0]):
            vid_tchw = x[b].permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]
            out_cthw = self.teacher_transform(vid_tchw)       # [C,T,Ht,Wt]
            outs.append(out_cthw)
        x_teacher = torch.stack(outs, dim=0)  # [B,C,T,Ht,Wt]

        # dtype 对齐到 teacher
        if self.teacher_model is not None:
            target_dtype = next(self.teacher_model.parameters()).dtype
            x_teacher = x_teacher.to(dtype=target_dtype)

        return x_teacher

    @torch.no_grad()
    def _extract_vfm_features(self, x: torch.Tensor):
        """
        x: raw video [B, 3, T, H, W]
        Returns: [B, num_vfm_tokens, D_teacher]
        """
        assert self.teacher_model is not None, "Teacher model is required for encoding."
        if next(self.teacher_model.parameters()).device != x.device:
            self.teacher_model.to(x.device)
        t_input = self._preprocess_for_vjepa2(x)
        # 1. 获取特征列表
        # 假设 teacher 返回 [feat_l8, feat_l16, feat_l24, feat_l32]
        t_out = self.teacher_model(t_input)

        if not isinstance(t_out, (list, tuple)):
            # 兼容性处理：如果只返回了一个 Tensor，包一层
            t_out = [t_out]

        # 确保只取需要的 Tensor（有些实现可能返回 tuple）
        feats_list = []
        for f in t_out:
            if isinstance(f, (tuple, list)):
                f = f[0]
            feats_list.append(f) # List of [B, N, D]
        normed_feats = []
        for i, feat in enumerate(feats_list):
            # feat: [B, N, 1280]
            normed_feats.append(self.teacher_layer_norms[i](feat))
        # 2. 拼接 (Concatenation)
        # [B, N, D] * 4 -> [B, N, 4*D]
        # 必须转 float 以便进行 LayerNorm 和 Linear 计算（避免 bf16 溢出或精度问题）
        cat_feats = torch.cat(normed_feats, dim=-1).float() 
        # 3. 线性映射融合 (Linear Projection)
        # [B, N, 4*D] -> [B, N, 1280]
        fused_feats = self.fusion_proj(cat_feats)
        return fused_feats






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
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t)
            self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))
        if self.use_encoder_patch_token_type_embed:
            encoder_patch_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_patch_token_type_embed.data.copy_(encoder_patch_token_type_embed)

        # Initialize encoder latent query embed
        if self.learned_encoder_latent_query_embed:
            if self.encoder_query_gaussian_init:
                # from timm vision_transformer.py
                # https://github.com/huggingface/pytorch-image-models/blob/70ccf00c95a2d78a166cca24ef6adbca46f47c2a/timm/models/vision_transformer.py#L495
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




        imagedec_token_embed = get_3d_sincos_pos_embed(768, 16, 8)
        imagedec_token_embed = torch.from_numpy(imagedec_token_embed).float().reshape(1, 2048, 768)
        self.imagedec_latent_pe.data.copy_(imagedec_token_embed)
        imagedec_latent_token_type_embed = torch.randn(1, 1, 768) * .02
        self.imagedec_latent_token_type_embed.data.copy_(imagedec_latent_token_type_embed)




        # initialize decoder patch query PE
        if self.learned_decoder_patch_query_embed:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_t))
            self.decoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.decoder_h_embed))
            self.decoder_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.decoder_w_embed))
            self.decoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.decoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.decoder_token_h, self.decoder_token_t)
            self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        if self.use_decoder_patch_query_token_type_embed:
            decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)



        # Zero-out output layers:
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



    def encode(self, x):
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]


        return {'encoded': vfm_feats}



    def unpatchify(self, x):
        c = 3
        pt = 2
        p = 16
        h = w = 16
        t = x.size(1) // (h * w)
        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        # z: (b, n, d)
        b = z.size(0)

        imagedec_token_embed = self.get_imagedec_latent_pe() 
        imagedec_token_embed = imagedec_token_embed.expand(b, -1, -1)
        x= self.dec_to_decimage(z) 
        #print(f"DEBUG: x shape: {x.shape}, imagedec_token_embed shape: {imagedec_token_embed.shape}") 
        if self.use_pe=='yes':
            x = x + imagedec_token_embed
        # = x + imagedec_token_embed
        #x= self.dec_to_decimage(x)
        x = self.decoder_args_dec_to_image(x)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        B = data.size(0)
        encode_output = self.encode(data)

        pred_frames = self.decode(encode_output['encoded'])
        pred_frames = pred_frames.contiguous()
        return_dict = {'pred_frames': pred_frames, **encode_output}


        return return_dict
