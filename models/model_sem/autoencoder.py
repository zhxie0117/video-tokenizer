import torch
import torch.nn as nn
from models.model_cnnvit.base.blocks import Encoder, Decoder ,Decoder_unify
from models.model_cnnvit.quantizer.fsq import FSQ
from models import register
import torch.nn.functional as F
from einops import rearrange





@register('autoencoder_cnnvit')
class AutoEncoder(nn.Module):
    def __init__(self, 
        bottleneck,
        prior_model,
        # --- Token 数量配置 ---
        num_latent_tokens=1024,    # 原本是 128+256，现在统一为一个总数
        # --- 尺寸配置 ---
        input_size=128,
        frame_num=16,             # 统一处理的帧数
        temporal_patch_size=4,    # 时间维度的 Patch Size bottleneck_token_num
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        # --- 模型配置 ---
        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        # --- Embedding 配置 ---
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_query_gaussian_init=True,
        
        # --- Boolean Flags ---
        learned_decoder_latent_pe=False,
        
        **kwargs):
        super().__init__()

        in_grid =  [16, 128, 128]
        token_size = 6

        self.encoder = Encoder(
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            out_grid=in_grid,
        )
        self.prior_model = None 
    def encode(self, data, **kwargs):
        x_first = data[:, :, 0:1, :, :]  # [B, C, 1, H, W]
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def decode_indices(self, indices):
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q)
    
    def forward(self, x):
        x_q, out_dict = self.encode(x)
        x = self.decode(x_q)
        return_dict = {'pred_frames': x.contiguous()}
        return return_dict
    










import sys
import os
import yaml
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from transformers import AutoConfig 

# 假设你的环境中有 jepa 源码
sys.path.append("jepa/")

try:
    from jepa.evals.video_classification_frozen.eval import init_model
except ImportError:
    print("Warning: 'jepa' module not found. Please ensure JEDi source code is in the path.")

# ==========================================
# 1. 工具函数
# ==========================================
VJEPA_CKPT_URLS = {
    'vit_large': {
        'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vitl16_ssv2_16x2x3.yaml',
        'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar',
    },
    'vit_huge': {
        'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vith16_ssv2_16x2x3.yaml',
        'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar',
    },
}

def download_file(url, local_path, chunk_size=1024):
    if os.path.exists(local_path):
        return
    print(f"Downloading {url} to {local_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open(local_path, "wb") as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

# ==========================================
# 2. 对齐模块 (Aligner)
# ==========================================
class VJepaAligner(nn.Module):
    def __init__(self, student_dim, teacher_dim, student_grid):
        super().__init__()
        self.student_grid = student_grid # (T_s, H_s, W_s) -> (4, 16, 16)
        
        # 投影层：把 Student 特征维度对齐到 Teacher
        self.projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

    def forward(self, student_q, teacher_feats, teacher_grid_shape):
        """
        student_q: [B, 1024, 6] (Student Latent)
        teacher_feats: [B, 1568, D] (Teacher Output)
        teacher_grid_shape: (8, 14, 14)
        """
        # 1. 投影 Student [B, 1024, 6] -> [B, 1024, D]
        student_proj = self.projector(student_q)
        
        # 2. 恢复 Student 的 3D 结构
        ts, hs, ws = self.student_grid # (4, 16, 16)
        # 假设 student_q 是按 (T H W) 顺序排列
        s_3d = rearrange(student_proj, 'b (t h w) c -> b c t h w', t=ts, h=hs, w=ws)
        
        # 3. 恢复 Teacher 的 3D 结构 (关键！)
        tt, ht, wt = teacher_grid_shape # (8, 14, 14)
        expected_tokens = tt * ht * wt  # 1568
        
        # 处理可能的 [CLS] token (如果 Teacher 输出 1569，第一个通常是 CLS)
        if teacher_feats.shape[1] == expected_tokens + 1:
            teacher_feats = teacher_feats[:, 1:, :] # 丢弃 CLS
        
        # 确保 Token 数量完全匹配再 Reshape，否则报错比隐式截断好
        assert teacher_feats.shape[1] == expected_tokens, \
            f"Token mismatch! Teacher: {teacher_feats.shape[1]}, Expected Grid {teacher_grid_shape}={expected_tokens}"

        # Reshape Teacher: [B, 1568, D] -> [B, D, 8, 14, 14]
        t_3d = rearrange(teacher_feats, 'b (t h w) c -> b c t h w', t=tt, h=ht, w=wt)
        
        # 4. 时空对齐 (Interpolation)
        # 将 Teacher 的 (8, 14, 14) 插值成 Student 的 (4, 16, 16)
        # 这一步解决了分辨率不一致和 Token 数不一致的问题
        t_aligned = F.interpolate(
            t_3d, 
            size=(ts, hs, ws),  # 目标尺寸 (4, 16, 16)
            mode='trilinear', 
            align_corners=False
        )
        #print(f"Aligned Teacher Shape: {t_aligned.shape}, Student Shape: {s_3d.shape}")
        # 5. 计算 Cosine Loss
        # 展平回 [B*Tokens, D]
        s_flat = rearrange(s_3d, 'b c t h w -> (b t h w) c')
        t_flat = rearrange(t_aligned, 'b c t h w -> (b t h w) c')
        
        # Normalize & Dot Product
        s_norm = F.normalize(s_flat, dim=-1, eps=1e-6)
        t_norm = F.normalize(t_flat, dim=-1, eps=1e-6)
        
        # Loss = 1 - CosSim
        loss = 1.0 - (s_norm * t_norm).sum(dim=-1).mean()
        
        return loss

# ==========================================
# 3. AutoEncoder (修正版)
# ==========================================
@register('autoencoder_cnnvit_align1')
class AutoEncoder1(nn.Module):
    def __init__(self, 
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=128,
        
        # --- V-JEPA Config ---
        use_vjepa_loss=True,
        vjepa_model_name='vit_large',
        vjepa_ckpt_dir='./checkpoints/jepa',
        
        **kwargs):
        super().__init__()

        self.latent_grid = (4, 16, 16) 
        in_grid = [16, 128, 128]
        token_size = 6

        # --- Student ---
        self.encoder = Encoder(
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=num_latent_tokens,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=num_latent_tokens,
            out_grid=in_grid,
        )
        
        # --- Teacher ---
        self.use_vjepa_loss = use_vjepa_loss
        self.teacher_model = None
        self.prior_model = None 
        if self.use_vjepa_loss:
            self._init_vjepa_teacher(vjepa_model_name, vjepa_ckpt_dir, token_size)

    def _init_vjepa_teacher(self, model_name, save_dir, student_dim):
        print(f"Initializing V-JEPA Teacher ({model_name})...")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 1. 下载
            if model_name not in VJEPA_CKPT_URLS:
                raise ValueError(f"Unknown model_name: {model_name}")
            urls = VJEPA_CKPT_URLS[model_name]
            model_path = os.path.join(save_dir, f"{model_name}.pth.tar")
            config_path = os.path.join(save_dir, f"{model_name}.yaml")
            download_file(urls['model_ckpt'], model_path)
            download_file(urls['config'], config_path)
            
            # 2. Config
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            self.teacher_patch_size = config['pretrain']['patch_size']
            self.teacher_tubelet_size = config['pretrain']['tubelet_size']
            self.teacher_resolution = config['optimization']['resolution']
            
            # 3. Init Model (直接使用 base model，不要 ClipAggregation)
            # 注意: 这里使用位置参数传递 model_path
            self.teacher_model = init_model(
                "cpu",
                model_path, 
                model_name,
                patch_size=self.teacher_patch_size,
                crop_size=self.teacher_resolution,
                frames_per_clip=config['pretrain']['frames_per_clip'],
                tubelet_size=self.teacher_tubelet_size,
                use_sdpa=config['pretrain'].get('use_sdpa', False),
                use_SiLU=config['pretrain'].get('use_silu', False),
                tight_SiLU=config['pretrain'].get('tight_silu', True),
                uniform_power=config['pretrain'].get('uniform_power', False),
            )
            print(f"V-JEPA Teacher Config: Patch Size: {self.teacher_patch_size}, Tubelet Size: {self.teacher_tubelet_size}, Resolution: {self.teacher_resolution}")
            print(config['pretrain']['frames_per_clip'])
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            teacher_dim = self.teacher_model.embed_dim
            print(f"Teacher loaded. Dim: {teacher_dim}, Res: {self.teacher_resolution}")
            self.aligner = VJepaAligner(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                student_grid=self.latent_grid
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: Failed to load V-JEPA Teacher: {e}")
            self.use_vjepa_loss = False

    def _preprocess_batch(self, x):
        B, C, T, H, W = x.shape
        x_flat = rearrange(x, 'b c t h w -> (b t) c h w')
        
        # Resize
        x_resized = F.interpolate(x_flat, size=(self.teacher_resolution, self.teacher_resolution), 
                                  mode='bicubic', align_corners=False)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_norm = (x_resized - mean) / std
        x_out = rearrange(x_norm, '(b t) c h w -> b c t h w', b=B, t=T)
        # Dtype match
        if self.teacher_model is not None:
            target_dtype = next(self.teacher_model.parameters()).dtype
            x_out = x_out.to(target_dtype)
            
        return x_out

    def encode(self, data, **kwargs):
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        # x: [B, 3, 16, 128, 128]
        
        x_q, out_dict = self.encode(x)
        pred_x = self.decode(x_q)
        return_dict = {'pred_frames': pred_x.contiguous()}
        
        if self.training and self.use_vjepa_loss and self.teacher_model is not None:
            # Move teacher to device if needed
            if next(self.teacher_model.parameters()).device != x.device:
                self.teacher_model.to(x.device)

            with torch.no_grad():
                t_input = self._preprocess_batch(x)
                t_feats = self.teacher_model(t_input) # Output: [B, N, D]
                t_feats = t_feats.float()

            tt = x.shape[2] // self.teacher_tubelet_size
            ht = self.teacher_resolution // self.teacher_patch_size
            wt = self.teacher_resolution // self.teacher_patch_size
            loss_align = self.aligner(x_q.float(), t_feats, (tt, ht, wt))
            return_dict['align_loss'] = loss_align
            
        return return_dict




# import sys
# import os
# import yaml
# import requests
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from einops import rearrange

# # 假设你的环境中有 jepa 源码
# sys.path.append("jepa/")

# try:
#     from jepa.evals.video_classification_frozen.eval import init_model
# except ImportError:
#     print("Warning: 'jepa' module not found. Please ensure JEDi source code is in the path.")

# # ==========================================
# # 1. 工具函数
# # ==========================================
# VJEPA_CKPT_URLS = {
#     'vit_large': {
#         'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vitl16_ssv2_16x2x3.yaml',
#         'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar',
#     },
#     'vit_huge': {
#         'config': 'https://raw.githubusercontent.com/facebookresearch/jepa/refs/heads/main/configs/evals/vith16_ssv2_16x2x3.yaml',
#         'model_ckpt': 'https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar',
#     },
# }

# def download_file(url, local_path, chunk_size=1024):
#     if os.path.exists(local_path):
#         return
#     print(f"Downloading {url} to {local_path}...")
#     try:
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status()
#             total_size = int(r.headers.get("content-length", 0))
#             with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
#                 with open(local_path, "wb") as f:
#                     for data in r.iter_content(chunk_size=chunk_size):
#                         if data:
#                             f.write(data)
#                             pbar.update(len(data))
#     except Exception as e:
#         print(f"Download failed: {e}")
#         if os.path.exists(local_path):
#             os.remove(local_path)

# # ==========================================
# # 2. 对齐模块：共同维度 + 聚类原型 + Gram + PCA 子空间
# # ==========================================
# class SoftKMeans(nn.Module):
#     """
#     可微“软 k-means”原型聚合：
#     输入 tokens: [B, N, D]
#     输出 prototypes: [B, K, D]
#     """
#     def __init__(self, num_prototypes=256, iters=5, temp=0.2, eps=1e-6):
#         super().__init__()
#         self.K = num_prototypes
#         self.iters = iters
#         self.temp = temp
#         self.eps = eps

#     @torch.cuda.amp.autocast(enabled=False)
#     def forward(self, x):
#         """
#         x: [B, N, D] float32 更稳
#         """
#         x = x.float()
#         B, N, D = x.shape

#         # init: 从 token 中随机取 K 个做初始中心（每个 batch 独立）
#         # 也可以用更稳定的策略：均匀采样/网格采样；这里保持简单
#         idx = torch.randint(0, N, (B, self.K), device=x.device)
#         c = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))  # [B,K,D]

#         for _ in range(self.iters):
#             # squared euclidean distance: ||x-c||^2 = x^2 + c^2 - 2 x·c
#             x2 = (x ** 2).sum(dim=-1, keepdim=True)          # [B,N,1]
#             c2 = (c ** 2).sum(dim=-1).unsqueeze(1)           # [B,1,K]
#             xc = torch.bmm(x, c.transpose(1, 2))             # [B,N,K]
#             dist2 = x2 + c2 - 2 * xc                         # [B,N,K]

#             # soft assignments over K (for each token)
#             w = torch.softmax(-dist2 / max(self.temp, self.eps), dim=-1)  # [B,N,K]

#             # update centers: c_k = sum_i w_{ik} x_i / sum_i w_{ik}
#             denom = w.sum(dim=1).unsqueeze(-1) + self.eps                 # [B,K,1]
#             c = torch.bmm(w.transpose(1, 2), x) / denom                    # [B,K,D]

#         return c


# def gram_matrix(tokens, normalize_tokens=True, eps=1e-6):
#     """
#     tokens: [B, K, D]
#     return: [B, K, K] Gram matrix
#     """
#     if normalize_tokens:
#         tokens = F.normalize(tokens, dim=-1, eps=eps)
#     return torch.bmm(tokens, tokens.transpose(1, 2))


# def pca_subspace_basis(tokens, r=32, center=True, eps=1e-6):
#     """
#     计算 token 集合的 PCA 子空间基底（前 r 个主方向）
#     tokens: [B, K, D]
#     return: basis [B, D, r]  (列正交，近似)
#     说明：用 SVD 求解，D 可能较大但 K=256 时开销可接受。
#     """
#     x = tokens
#     if center:
#         x = x - x.mean(dim=1, keepdim=True)
#     # x: [B,K,D] -> 做每个 batch 的 SVD
#     # 我们需要右奇异向量 V: [B, D, D]，取前 r 列
#     # torch.linalg.svd 支持 batch
#     # 注意：full_matrices=False 更快
#     U, S, Vh = torch.linalg.svd(x, full_matrices=False)  # Vh: [B, D, D] (或 [B, D, D'] 取决于 K,D)
#     # 这里 x 的形状是 KxD，full_matrices=False => Vh: [B, min(K,D), D]
#     # 因为 K=256，D=common_dim（例如 512），min=256，所以 Vh: [B, 256, D]
#     # 右奇异向量在 Vh 的行里，因此主方向是 V = Vh^T 的前 r 列
#     V = Vh.transpose(-2, -1)  # [B, D, min(K,D)]
#     r_eff = min(r, V.shape[-1])
#     basis = V[:, :, :r_eff]   # [B, D, r_eff]
#     return basis


# def subspace_alignment_loss(U_tokens, V_tokens, r=32):
#     """
#     子空间对齐：最大化主子空间相似度 (projection Frobenius)
#     令 B_u, B_v 为 [B, D, r] 的正交基底，loss = r - ||B_u^T B_v||_F^2
#     """
#     Bu = pca_subspace_basis(U_tokens, r=r)  # [B,D,r]
#     Bv = pca_subspace_basis(V_tokens, r=r)  # [B,D,r]
#     M = torch.bmm(Bu.transpose(1, 2), Bv)   # [B,r,r]
#     sim = (M ** 2).sum(dim=(1, 2))          # [B]
#     r_eff = M.shape[1]
#     loss = (r_eff - sim).mean()
#     return loss


# class VJepaAlignerV3(nn.Module):
#     """
#     改造版对齐：
#     1) Student/Teacher 都投影到共同维度 common_dim
#     2) 对齐 Teacher 到 Student 的时空网格（保持你原来的插值逻辑）
#     3) 两边各自用 SoftKMeans 聚合到 256 个 prototypes
#     4) Gram loss 对齐关系结构
#     5) PCA 子空间对齐作为全局约束
#     """
#     def __init__(
#         self,
#         student_dim,
#         teacher_dim,
#         student_grid,
#         common_dim=512,
#         num_prototypes=256,
#         kmeans_iters=5,
#         kmeans_temp=0.2,
#         gram_weight=1.0,
#         pca_weight=0.2,
#         pca_rank=32,
#     ):
#         super().__init__()
#         self.student_grid = student_grid
#         self.common_dim = common_dim
#         self.num_prototypes = num_prototypes

#         # 两边投影到共同 emb dim
#         self.student_proj = nn.Sequential(
#             nn.Linear(student_dim, common_dim),
#             nn.LayerNorm(common_dim),
#             nn.GELU(),
#             nn.Linear(common_dim, common_dim),
#         )
#         self.teacher_proj = nn.Sequential(
#             nn.Linear(teacher_dim, common_dim),
#             nn.LayerNorm(common_dim),
#             nn.GELU(),
#             nn.Linear(common_dim, common_dim),
#         )

#         self.pool = SoftKMeans(num_prototypes=num_prototypes, iters=kmeans_iters, temp=kmeans_temp)

#         self.gram_weight = gram_weight
#         self.pca_weight = pca_weight
#         self.pca_rank = pca_rank

#     def forward(self, student_q, teacher_feats, teacher_grid_shape):
#         """
#         student_q:   [B, 1024, student_dim]
#         teacher_feats:[B, 1568 or 1569, teacher_dim]
#         teacher_grid_shape: (tt, ht, wt) e.g. (8,14,14)
#         """
#         B = student_q.shape[0]

#         # 1) Project both to common_dim
#         s = self.student_proj(student_q)      # [B,1024,Dc]
#         t = self.teacher_proj(teacher_feats)  # [B,Nt,Dc]

#         # 2) reshape student -> 3D grid
#         ts, hs, ws = self.student_grid
#         s_3d = rearrange(s, 'b (t h w) c -> b c t h w', t=ts, h=hs, w=ws)

#         # 3) reshape teacher -> 3D grid (drop CLS if exists)
#         tt, ht, wt = teacher_grid_shape
#         expected_tokens = tt * ht * wt

#         if t.shape[1] == expected_tokens + 1:
#             t = t[:, 1:, :]  # drop CLS

#         assert t.shape[1] == expected_tokens, \
#             f"Token mismatch! Teacher: {t.shape[1]}, Expected Grid {teacher_grid_shape}={expected_tokens}"

#         t_3d = rearrange(t, 'b (t h w) c -> b c t h w', t=tt, h=ht, w=wt)

#         # 4) align teacher grid to student grid via trilinear interpolation
#         t_aligned = F.interpolate(
#             t_3d,
#             size=(ts, hs, ws),
#             mode='trilinear',
#             align_corners=False
#         )

#         # 5) flatten back to tokens: [B, Ns, Dc]
#         s_tok = rearrange(s_3d, 'b c t h w -> b (t h w) c')       # [B,1024,Dc]
#         t_tok = rearrange(t_aligned, 'b c t h w -> b (t h w) c')  # [B,1024,Dc]

#         # 6) prototype pooling to K=256 via soft kmeans
#         # 为了更稳，先 normalize 再聚类（可选）
#         s_tok_n = F.normalize(s_tok, dim=-1, eps=1e-6)
#         t_tok_n = F.normalize(t_tok, dim=-1, eps=1e-6)
#         s_proto = self.pool(s_tok_n)  # [B,256,Dc]
#         t_proto = self.pool(t_tok_n)  # [B,256,Dc]

#         # 7) Gram loss: align relational structure among prototypes
#         Gs = gram_matrix(s_proto, normalize_tokens=True)  # [B,256,256]
#         Gt = gram_matrix(t_proto, normalize_tokens=True)
#         gram_loss = F.mse_loss(Gs, Gt)

#         # 8) PCA subspace alignment (global constraint)
#         pca_loss = subspace_alignment_loss(s_proto, t_proto, r=self.pca_rank)
#         print(f"Gram Loss: {gram_loss.item():.4f}, PCA Loss: {pca_loss.item():.4f}")
#         loss = self.gram_weight * gram_loss + self.pca_weight * pca_loss
#         return loss, {"gram_loss": gram_loss.detach(), "pca_loss": pca_loss.detach()}

# # ==========================================
# # 3. AutoEncoder (改造版：使用新 aligner)
# # ==========================================
# # 你的工程里应该已经有 register / Encoder / Decoder / FSQ
# # 这里保持你的结构不变，只替换 aligner 与 loss 输出逻辑
# @register('autoencoder_cnnvit_softalign')
# class AutoEncoder1(nn.Module):
#     def __init__(
#         self,
#         bottleneck,
#         prior_model,
#         num_latent_tokens=1024,
#         input_size=128,

#         # --- V-JEPA Config ---
#         use_vjepa_loss=True,
#         vjepa_model_name='vit_large',
#         vjepa_ckpt_dir='./checkpoints/jepa',

#         # --- New alignment config ---
#         align_common_dim=512,
#         align_num_prototypes=256,
#         align_kmeans_iters=5,
#         align_kmeans_temp=0.2,
#         align_gram_weight=1.0,
#         align_pca_weight=0.2,
#         align_pca_rank=32,

#         **kwargs
#     ):
#         super().__init__()

#         self.latent_grid = (4, 16, 16)
#         in_grid = [16, 128, 128]
#         token_size = 6

#         # --- Student ---
#         self.encoder = Encoder(
#             model_size='small_thin',
#             patch_size=[4, 8, 8],
#             in_channels=3,
#             out_channels=token_size,
#             in_grid=in_grid,
#             out_tokens=num_latent_tokens,
#         )
#         self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
#         self.decoder = Decoder(
#             model_size='small_thin',
#             patch_size=[4, 8, 8],
#             in_channels=token_size,
#             out_channels=3,
#             in_tokens=num_latent_tokens,
#             out_grid=in_grid,
#         )

#         # --- Teacher ---
#         self.use_vjepa_loss = use_vjepa_loss
#         self.teacher_model = None
#         self.prior_model = None

#         # save align cfg
#         self.align_common_dim = align_common_dim
#         self.align_num_prototypes = align_num_prototypes
#         self.align_kmeans_iters = align_kmeans_iters
#         self.align_kmeans_temp = align_kmeans_temp
#         self.align_gram_weight = align_gram_weight
#         self.align_pca_weight = align_pca_weight
#         self.align_pca_rank = align_pca_rank

#         if self.use_vjepa_loss:
#             self._init_vjepa_teacher(vjepa_model_name, vjepa_ckpt_dir, token_size)

#     def _init_vjepa_teacher(self, model_name, save_dir, student_dim):
#         print(f"Initializing V-JEPA Teacher ({model_name})...")
#         os.makedirs(save_dir, exist_ok=True)

#         try:
#             # 1) download
#             if model_name not in VJEPA_CKPT_URLS:
#                 raise ValueError(f"Unknown model_name: {model_name}")
#             urls = VJEPA_CKPT_URLS[model_name]
#             model_path = os.path.join(save_dir, f"{model_name}.pth.tar")
#             config_path = os.path.join(save_dir, f"{model_name}.yaml")
#             download_file(urls['model_ckpt'], model_path)
#             download_file(urls['config'], config_path)

#             # 2) load config
#             with open(config_path, 'r') as f:
#                 config = yaml.load(f, Loader=yaml.FullLoader)

#             self.teacher_patch_size = config['pretrain']['patch_size']
#             self.teacher_tubelet_size = config['pretrain']['tubelet_size']
#             self.teacher_resolution = config['optimization']['resolution']

#             # 3) init teacher model
#             self.teacher_model = init_model(
#                 "cpu",
#                 model_path,
#                 model_name,
#                 patch_size=self.teacher_patch_size,
#                 crop_size=self.teacher_resolution,
#                 frames_per_clip=config['pretrain']['frames_per_clip'],
#                 tubelet_size=self.teacher_tubelet_size,
#                 use_sdpa=config['pretrain'].get('use_sdpa', False),
#                 use_SiLU=config['pretrain'].get('use_silu', False),
#                 tight_SiLU=config['pretrain'].get('tight_silu', True),
#                 uniform_power=config['pretrain'].get('uniform_power', False),
#             )
#             print(f"V-JEPA Teacher Config: Patch Size: {self.teacher_patch_size}, Tubelet Size: {self.teacher_tubelet_size}, Resolution: {self.teacher_resolution}")
#             print(config['pretrain']['frames_per_clip'])

#             self.teacher_model.eval()
#             for p in self.teacher_model.parameters():
#                 p.requires_grad = False

#             teacher_dim = self.teacher_model.embed_dim
#             print(f"Teacher loaded. Dim: {teacher_dim}, Res: {self.teacher_resolution}")

#             # 4) new aligner
#             self.aligner = VJepaAlignerV3(
#                 student_dim=student_dim,
#                 teacher_dim=teacher_dim,
#                 student_grid=self.latent_grid,
#                 common_dim=self.align_common_dim,
#                 num_prototypes=self.align_num_prototypes,
#                 kmeans_iters=self.align_kmeans_iters,
#                 kmeans_temp=self.align_kmeans_temp,
#                 gram_weight=self.align_gram_weight,
#                 pca_weight=self.align_pca_weight,
#                 pca_rank=self.align_pca_rank,
#             )

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             print(f"ERROR: Failed to load V-JEPA Teacher: {e}")
#             self.use_vjepa_loss = False

#     def _preprocess_batch(self, x):
#         """
#         x: [B, 3, T, H, W]
#         resize + imagenet normalize, and dtype match teacher
#         """
#         B, C, T, H, W = x.shape
#         x_flat = rearrange(x, 'b c t h w -> (b t) c h w')

#         x_resized = F.interpolate(
#             x_flat,
#             size=(self.teacher_resolution, self.teacher_resolution),
#             mode='bicubic',
#             align_corners=False
#         )

#         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
#         x_norm = (x_resized - mean) / std

#         x_out = rearrange(x_norm, '(b t) c h w -> b c t h w', b=B, t=T)

#         if self.teacher_model is not None:
#             target_dtype = next(self.teacher_model.parameters()).dtype
#             x_out = x_out.to(target_dtype)

#         return x_out

#     def encode(self, data, **kwargs):
#         x = self.encoder(data)
#         x_q, x_dict = self.quantize(x)
#         return x_q, x_dict

#     def decode(self, x):
#         return self.decoder(x)

#     def forward(self, x):
#         # x: [B, 3, 16, 128, 128]
#         x_q, out_dict = self.encode(x)
#         pred_x = self.decode(x_q)

#         return_dict = {'pred_frames': pred_x.contiguous()}

#         if self.training and self.use_vjepa_loss and (self.teacher_model is not None):
#             # Move teacher to device if needed
#             if next(self.teacher_model.parameters()).device != x.device:
#                 self.teacher_model.to(x.device)

#             with torch.no_grad():
#                 t_input = self._preprocess_batch(x)
#                 t_feats = self.teacher_model(t_input)  # [B, Nt, Dt]
#                 t_feats = t_feats.float()

#             tt = x.shape[2] // self.teacher_tubelet_size
#             ht = self.teacher_resolution // self.teacher_patch_size
#             wt = self.teacher_resolution // self.teacher_patch_size

#             align_loss, align_dict = self.aligner(x_q.float(), t_feats, (tt, ht, wt))
#             return_dict['align_loss'] = align_loss
#             return_dict.update({f"align_{k}": v for k, v in align_dict.items()})

#         return return_dict

























































