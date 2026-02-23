
import torch
import torch.nn as nn
from models.model_sem.base.blocks import TokenizerEncoder1D, TokenizerDecoder1D, VideoDecoder
from models.model_sem.quantizer.fsq import FSQ
from models import register
import torch.nn.functional as F
from einops import rearrange

import os
import sys
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vjepa2.src.models.vision_transformer import vit_huge_rope
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
    #short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(256, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

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

# class SemanticInjector(nn.Module):
#     """
#     语义注入模块 (Semantic Injector Block)
#     作用：利用深层语义特征 (Semantic/Deep) 来调制浅层细节特征 (Detail/Shallow)。
#     实现：SPADE/AdaIN 的 3D 卷积变体。
#     """
#     def __init__(self, dim, kernel_size=3):
#         super().__init__()
#         self.dim = dim
        
#         # 浅层特征的标准化 (类似于 SPADE 之前的 Norm)
#         # GroupNorm 在 Batch Size 较小时比 LayerNorm/BatchNorm 在 3D 上更稳定
#         self.norm_shallow = nn.GroupNorm(32, dim)

#         # 调制参数生成器 (从深层特征生成)
#         # 这是一个轻量级的 3D 卷积网络
#         padding = kernel_size // 2
#         self.modulation_generator = nn.Sequential(
#             nn.SiLU(),
#             nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding),
#             nn.SiLU(),
#             nn.Conv3d(dim, 2 * dim, kernel_size=kernel_size, padding=padding) 
#         )
        
#         # 零初始化：这是为了训练稳定。初始状态下，scale=0, shift=0
#         # 这样初始输出 approx equal to 原始 shallow feature，避免破坏预训练特征分布
#         nn.init.constant_(self.modulation_generator[-1].weight, 0)
#         nn.init.constant_(self.modulation_generator[-1].bias, 0)

#     def forward(self, x_shallow, x_deep, shape_info):
#         """
#         x_shallow: [B, N, D] - 待调制的细节特征
#         x_deep:    [B, N, D] - 提供语义的指导特征
#         shape_info: (T, H, W) - 用于还原 3D 结构
#         """
#         B, N, D = x_shallow.shape
#         T, H, W = shape_info
        
#         # 1. 还原 3D 结构 [B, D, T, H, W] 以利用空间局部性
#         shallow_3d = x_shallow.transpose(1, 2).reshape(B, D, T, H, W)
#         deep_3d = x_deep.transpose(1, 2).reshape(B, D, T, H, W)

#         # 2. 计算 shallow 的标准化特征
#         shallow_norm = self.norm_shallow(shallow_3d)

#         # 3. 生成调制参数 (Scale & Shift)
#         # style_feats: [B, 2*D, T, H, W]
#         style_feats = self.modulation_generator(deep_3d)
#         scale, shift = style_feats.chunk(2, dim=1) 
        
#         # scale 需要加 1，因为初始希望是乘以 1 (即不改变)
#         scale = scale + 1.0

#         # 4. 注入 (Modulate)
#         # out = scale * norm(content) + shift
#         out_3d = shallow_norm * scale + shift
        
#         # 5. 残差连接 (这是可选的，但建议加上以保留原始细节流的梯度)
#         out_3d = out_3d + shallow_3d

#         # 6. 展平回 Sequence [B, N, D]
#         return out_3d.flatten(2).transpose(1, 2)


# class SemanticPyramidFusion(nn.Module):
#     """
#     语义金字塔融合 (Semantic Pyramid Fusion)
#     替代之前的 GatedLinearLayerFusion。
#     策略：Top-Down Pathway。
#     L32 (Deepest) -> Modulate L24 -> Modulate L16 -> Modulate L8 (Result)
#     """
#     def __init__(self, dim, grid_size):
#         super().__init__()
#         self.dim = dim
#         self.grid_size = grid_size  # (T, H, W)
        
#         # 定义 3 个注入层
#         # Injector 1: L32 -> L24
#         self.injector_l24 = SemanticInjector(dim)
#         # Injector 2: (L32+L24) -> L16
#         self.injector_l16 = SemanticInjector(dim)
#         # Injector 3: (L32+L24+L16) -> L8
#         self.injector_l8 = SemanticInjector(dim)
        
#         # 最终融合后的处理，将多尺度的信息平滑一下
#         self.final_proj = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim)
#         )

#     def forward(self, feats_list):
#         """
#         feats_list: [L8, L16, L24, L32]  (假设输入顺序是浅层到深层)
#         注意：VFM提取的特征列表通常是按层号递增排序的
#         """
#         assert len(feats_list) == 4, "Expected 4 layers [L8, L16, L24, L32]"
        
#         f_l8, f_l16, f_l24, f_l32 = feats_list
        
#         # --- Top-Down Pathway ---
        
#         # Step 1: 用 L32 调制 L24
#         # 语义流 (current_sem) 初始为 L32
#         # 细节流 (target) 为 L24
#         feat_flow = self.injector_l24(x_shallow=f_l24, x_deep=f_l32, shape_info=self.grid_size)
        
#         # Step 2: 用 (Step 1的结果) 调制 L16
#         feat_flow = self.injector_l16(x_shallow=f_l16, x_deep=feat_flow, shape_info=self.grid_size)
        
#         # Step 3: 用 (Step 2的结果) 调制 L8
#         # 此时 f_l8 将包含最丰富的纹理细节，同时被深层语义矫正过
#         feat_flow = self.injector_l8(x_shallow=f_l8, x_deep=feat_flow, shape_info=self.grid_size)
        
#         # Final Projection
#         out = self.final_proj(feat_flow)
        
#         return out


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



@register("autoencoder_vfm")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,

        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='base_thin',
            in_channels=1280,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])

        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='base_thin',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=2048,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='large',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=2048,
            token_grid=self.vfm_grid,
            patch_size=[2, 16, 16],  # 与 tokenizer encoder patch size 对齐（即 ViT tubelet/patch size）
        )
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher(student_dim=token_size)
        self.aligner= nn.Linear(decoder_1d_width, 1280) 
        self.dicklinear= nn.Linear(1280, decoder_1d_width)
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(1280*4),      # 关键：先归一化，平衡不同层的数值范围
            nn.Linear(1280*4, 1280), # 投影回 1280
            nn.GELU()                      # 可选：增加非线性，通常有助于特征混合
        )
        # self.vjepa2_fuser = GatedLinearLayerFusion(
        #     dim=1280,  # default teacher dim; will adjust to teacher.embed_dim at init
        #     num_layers=4,

        # )

    def _init_vjepa2_teacher(self, student_dim: int):
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

        # 2. 拼接 (Concatenation)
        # [B, N, D] * 4 -> [B, N, 4*D]
        # 必须转 float 以便进行 LayerNorm 和 Linear 计算（避免 bf16 溢出或精度问题）
        cat_feats = torch.cat(feats_list, dim=-1).float() 

        # 3. 线性映射融合 (Linear Projection)
        # [B, N, 4*D] -> [B, N, 1280]
        fused_feats = self.fusion_proj(cat_feats)

        return fused_feats

        # t_feats = self.teacher_model(t_input)
        # fused = self.vjepa2_fuser(t_feats)  # [B,N,D]
        # return fused.float()

    def encode(self, x, **kwargs):
        vfm_feats = self._extract_vfm_features(x)           # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)           # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)              # [B, 1024, token_size]
        return latent_q, q_dict

    def decode(self, x_q):
        decoded_feats = self.tokenizer_decoder(x_q)           # [B, 2048, width]
        pred_video = self.video_decoder(decoded_feats)         # [B, 3, T, H, W]
        return pred_video

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        # --- VFM feature extraction (frozen teacher) ---
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]

        # --- 1D Tokenizer encode → quantize ---
        latent = self.tokenizer_encoder(vfm_feats)             # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)                # [B, 1024, token_size]

        # --- 1D Tokenizer decode → video decode ---
        decoded_feats = self.tokenizer_decoder(latent_q)        # [B, 2048, width]
        #decoded_feats=self.dicklinear(vfm_feats)#test video decoder with teacher features, should be 2048→decoder_1d_width
        pred_video = self.video_decoder(decoded_feats)          # [B, 3, 16, 128, 128]
        return_dict = {"pred_frames": pred_video.contiguous()}

        align_decoder_feats = self.aligner(decoded_feats.float())  # [B, 2048, 1024]
        align_loss= F.mse_loss(align_decoder_feats, vfm_feats.float())  # 对齐损失：decoder_feats vs teacher_feats
        return_dict["align_loss"] = 0.5 * align_loss

        return return_dict
    





@register("autoencoder_vfm2")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,

        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='small',
            in_channels=1280,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])

        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='small',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=2048,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='base',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=2048,
            token_grid=self.vfm_grid,
            patch_size=[2, 16, 16],  # 与 tokenizer encoder patch size 对齐（即 ViT tubelet/patch size）
        )
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher(student_dim=token_size)
        self.aligner= nn.Linear(decoder_1d_width, 1280) 
        self.dicklinear= nn.Linear(1280, decoder_1d_width)
        self.vjepa2_fuser = SemanticPyramidFusion(
            dim=1280,
            grid_size=self.vfm_grid
        )

    def _init_vjepa2_teacher(self, student_dim: int):
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

        fused = self.vjepa2_fuser(feats_list)

        return fused

        # t_feats = self.teacher_model(t_input)
        # fused = self.vjepa2_fuser(t_feats)  # [B,N,D]
        # return fused.float()

    def encode(self, x, **kwargs):
        vfm_feats = self._extract_vfm_features(x)           # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)           # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)              # [B, 1024, token_size]
        return latent_q, q_dict

    def decode(self, x_q):
        decoded_feats = self.tokenizer_decoder(x_q)           # [B, 2048, width]
        pred_video = self.video_decoder(decoded_feats)         # [B, 3, T, H, W]
        return pred_video

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        # --- VFM feature extraction (frozen teacher) ---
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]

        # --- 1D Tokenizer encode → quantize ---
        latent = self.tokenizer_encoder(vfm_feats)             # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)                # [B, 1024, token_size]

        # --- 1D Tokenizer decode → video decode ---
        decoded_feats = self.tokenizer_decoder(latent_q)        # [B, 2048, width]


        #decoded_feats=self.dicklinear(vfm_feats)#test video decoder with teacher features, should be 2048→decoder_1d_width
        pred_video = self.video_decoder(decoded_feats)          # [B, 3, 16, 128, 128]
        return_dict = {"pred_frames": pred_video.contiguous()}

        align_decoder_feats = self.aligner(decoded_feats.float())  # [B, 2048, 1024]
        align_loss= F.mse_loss(align_decoder_feats, vfm_feats.float())  # 对齐损失：decoder_feats vs teacher_feats
        return_dict["align_loss"] = 0.5 * align_loss

        return return_dict





@register("autoencoder_vfm1")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,
        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='base_thin',
            in_channels=1280,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='base_thin',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=2048,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='large',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=2048,
            token_grid=self.vfm_grid,
            patch_size=[2, 16, 16],  # 与 tokenizer encoder patch size 对齐（即 ViT tubelet/patch size）
        )
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher(student_dim=token_size)
        self.aligner= nn.Linear(decoder_1d_width, 1280) 
        self.dicklinear= nn.Linear(1280, decoder_1d_width)
        self.vjepa2_fuser = GatedLinearLayerFusion(
            dim=1280,  # default teacher dim; will adjust to teacher.embed_dim at init
            num_layers=4,
        )
    def _init_vjepa2_teacher(self, student_dim: int):
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
        t_feats = self.teacher_model(t_input)
        fused = self.vjepa2_fuser(t_feats)  # [B,N,D]
        return fused.float()

    def encode(self, x, **kwargs):
        vfm_feats = self._extract_vfm_features(x)           # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)           # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)              # [B, 1024, token_size]
        return latent_q, q_dict

    def decode(self, x_q):
        decoded_feats = self.tokenizer_decoder(x_q)           # [B, 2048, width]
        pred_video = self.video_decoder(decoded_feats)         # [B, 3, T, H, W]
        return pred_video

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        # --- VFM feature extraction (frozen teacher) ---
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]

        # --- 1D Tokenizer encode → quantize ---
        latent = self.tokenizer_encoder(vfm_feats)             # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)                # [B, 1024, token_size]
        # --- 1D Tokenizer decode → video decode ---
        decoded_feats = self.tokenizer_decoder(latent_q)        # [B, 2048, width]
        #decoded_feats=self.dicklinear(vfm_feats)#test video decoder with teacher features, should be 2048→decoder_1d_width
        pred_video = self.video_decoder(decoded_feats)          # [B, 3, 16, 128, 128]
        return_dict = {"pred_frames": pred_video.contiguous()}

        align_decoder_feats = self.aligner(decoded_feats.float())  # [B, 2048, 1024]
        align_loss= F.mse_loss(align_decoder_feats, vfm_feats.float())  # 对齐损失：decoder_feats vs teacher_feats
        return_dict["align_loss"] = 0.2 * align_loss

        return return_dict
    


@register("autoencoder_vfm_fianllayer")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,
        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='base_thin',
            in_channels=1280,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='base_thin',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=2048,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='large',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=2048,
            token_grid=self.vfm_grid,
            patch_size=[2, 16, 16],  # 与 tokenizer encoder patch size 对齐（即 ViT tubelet/patch size）
        )
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher(student_dim=token_size)
        self.aligner= nn.Linear(decoder_1d_width, 1280) 
        #self.dicklinear= nn.Linear(1280, decoder_1d_width)

    def _init_vjepa2_teacher(self, student_dim: int):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return
        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
        self.teacher_model = vit_huge_rope(
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
        t_feats = self.teacher_model(t_input)
        return t_feats.float()

    def encode(self, x, **kwargs):
        vfm_feats = self._extract_vfm_features(x)           # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)           # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)              # [B, 1024, token_size]
        return latent_q, q_dict

    def decode(self, x_q):
        decoded_feats = self.tokenizer_decoder(x_q)           # [B, 2048, width]
        pred_video = self.video_decoder(decoded_feats)         # [B, 3, T, H, W]
        return pred_video

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        # --- VFM feature extraction (frozen teacher) ---
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]
        # --- 1D Tokenizer encode → quantize ---
        latent = self.tokenizer_encoder(vfm_feats)             # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)                # [B, 1024, token_size]
        # --- 1D Tokenizer decode → video decode ---
        decoded_feats = self.tokenizer_decoder(latent_q)        # [B, 2048, width]
        #decoded_feats=self.dicklinear(vfm_feats)#test video decoder with teacher features, should be 2048→decoder_1d_width
        pred_video = self.video_decoder(decoded_feats)          # [B, 3, 16, 128, 128]
        return_dict = {"pred_frames": pred_video.contiguous()}
        align_decoder_feats = self.aligner(decoded_feats.float())  # [B, 2048, 1024]
        align_loss= F.mse_loss(align_decoder_feats, vfm_feats.float())  # 对齐损失：decoder_feats vs teacher_feats
        return_dict["align_loss"] = 0.2 * align_loss
        return return_dict
    


@register("autoencoder_vfm_fianllayer_noquant")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,
        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vith.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='base_thin',
            in_channels=1280,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='base_thin',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=2048,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='large',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=2048,
            token_grid=self.vfm_grid,
            patch_size=[2, 16, 16],  # 与 tokenizer encoder patch size 对齐（即 ViT tubelet/patch size）
        )
        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher(student_dim=token_size)
        self.aligner= nn.Linear(decoder_1d_width, 1280) 
        self.dicklinear= nn.Linear(1280, decoder_1d_width)

    def _init_vjepa2_teacher(self, student_dim: int):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return
        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
        self.teacher_model = vit_huge_rope(
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
        t_feats = self.teacher_model(t_input)
        return t_feats.float()

    def encode(self, x, **kwargs):
        vfm_feats = self._extract_vfm_features(x)           # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)           # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)              # [B, 1024, token_size]
        return latent_q, q_dict

    def decode(self, x_q):
        decoded_feats = self.tokenizer_decoder(x_q)           # [B, 2048, width]
        pred_video = self.video_decoder(decoded_feats)         # [B, 3, T, H, W]
        return pred_video

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        # --- VFM feature extraction (frozen teacher) ---
        vfm_feats = self._extract_vfm_features(x)             # [B, 2048, D_teacher]
        decoded_feats=self.dicklinear(vfm_feats)#test video decoder with teacher features, should be 2048→decoder_1d_width
        pred_video = self.video_decoder(decoded_feats)          # [B, 3, 16, 128, 128]
        return_dict = {"pred_frames": pred_video.contiguous()}

        return return_dict