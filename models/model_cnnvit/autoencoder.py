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
            model_size='small_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='small_thin',
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
@register('autoencoder_cnnvit_align')
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
            model_size='small_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=num_latent_tokens,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='small_thin',
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