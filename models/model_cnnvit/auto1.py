# -*- coding: utf-8 -*-
"""
将你现有的“语义对齐 teacher”从旧版 V-JEPA (jepa.evals...) 替换为 VJEPA2（纯 PyTorch，本地权重加载）。
不使用 HuggingFace 推理路径。

你需要保证工程内存在（来自 VJEPA2 仓库/你粘贴的 demo 同源代码）：
- src.models.vision_transformer.vit_giant_xformers_rope
- src.datasets.utils.video.transforms / volume_transforms
- （可选）src.models.attentive_pooler.AttentiveClassifier 仅分类时用，这里对齐不需要
并且准备好 VJEPA2 encoder 的本地权重 pt_model_path（下面参数 vjepa2_encoder_ckpt）。

输入约定：
- student 输入 x: [B, 3, T, H, W]，你这里 T=16, H=W=128
- teacher(VJEPA2) 输入: clip tensor [B, C, T', H', W']，T'由采样策略决定（默认64帧）
- teacher 输出 token: [B, Nt(+1?), Dt]（有的实现含 CLS，有的不含；下面都兼容）
"""
import torch
import torch.nn as nn
from models.model_cnnvit.base.blocks import Encoder, Decoder ,Decoder_unify
from models.model_cnnvit.quantizer.fsq import FSQ
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

# =========================================================
# 0) 引入 VJEPA2 的 PyTorch 实现（来自你贴的 demo 代码同源）
# =========================================================
# 需要你的工程里可 import 到这些模块
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


class VJepaAlignerV2(nn.Module):
    def __init__(
        self,
        student_dim,
        teacher_dim,
        student_grid,
        common_dim=512,
        num_prototypes=256,
        kmeans_iters=5,
        kmeans_temp=0.5,
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
            raise AssertionError(
                f"Token mismatch: teacher={t.shape[1]}, expected={expected_tokens} from grid {teacher_grid_shape}"
            )

        t_3d = rearrange(t, "b (t h w) c -> b c t h w", t=tt, h=ht, w=wt)

        # align teacher grid -> student grid
        t_aligned = F.interpolate(t_3d, size=(ts, hs, ws), mode="trilinear", align_corners=False)

        s_tok = rearrange(s_3d, "b c t h w -> b (t h w) c")
        t_tok = rearrange(t_aligned, "b c t h w -> b (t h w) c")

        # s_tok_n = F.normalize(s_tok, dim=-1, eps=1e-6)
        # t_tok_n = F.normalize(t_tok, dim=-1, eps=1e-6)
        s_proto = self.pool(s_tok)
        t_proto = self.pool(t_tok)


        Gs = gram_matrix(s_proto, normalize_tokens=True)
        Gt = gram_matrix(t_proto, normalize_tokens=True)
        gram_loss = F.mse_loss(Gs, Gt)
        vic_loss, vic_parts = vicreg_pooled_loss(
            s_tok, t_tok,
        )
        #print(f"Gram loss: {gram_loss.item():.4f}, PCA subspace loss: {pca_loss.item():.4f}")
        loss = self.gram_weight * gram_loss + 0.01 * vic_loss
        return loss, {"gram_loss": gram_loss.detach()}




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
            raise AssertionError(
                f"Token mismatch: teacher={t.shape[1]}, expected={expected_tokens} from grid {teacher_grid_shape}"
            )

        t_3d = rearrange(t, "b (t h w) c -> b c t h w", t=tt, h=ht, w=wt)

        # align teacher grid -> student grid
        t_aligned = F.interpolate(t_3d, size=(ts, hs, ws), mode="trilinear", align_corners=False)

        s_tok = rearrange(s_3d, "b c t h w -> b (t h w) c")
        t_tok = rearrange(t_aligned, "b c t h w -> b (t h w) c")

        # s_tok_n = F.normalize(s_tok, dim=-1, eps=1e-6)
        # t_tok_n = F.normalize(t_tok, dim=-1, eps=1e-6)
        s_proto = self.pool(s_tok)
        t_proto = self.pool(t_tok)



        # Gs = gram_matrix(s_proto, normalize_tokens=True)
        # Gt = gram_matrix(t_proto, normalize_tokens=True)


        gram_loss = F.mse_loss(s_proto, t_proto)
        #print(f"Gram loss: {gram_loss.item():.4f}, PCA subspace loss: {pca_loss.item():.4f}")
        loss = self.gram_weight * gram_loss
        return loss, {"gram_loss": gram_loss.detach()}


@register("autoencoder_cnnvit_softalign_gramonly_vjepa2")
class AutoEncoder1(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=128,

        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vitl.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        # --- Alignment config ---
        align_common_dim=256,
        align_num_prototypes=256,
        align_kmeans_iters=5,
        align_kmeans_temp=0.2,
        align_gram_weight=1.0,
        align_pca_weight=0.2,
        align_pca_rank=32,

        **kwargs,
    ):
        super().__init__()

        self.latent_grid = (4, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        in_grid = [16, 128, 128]
        token_size = 6  # student_dim

        # --- Student ---
        self.encoder = Encoder(
            model_size="small",
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=num_latent_tokens,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.decoder = Decoder(
            model_size="small",
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=num_latent_tokens,
            out_grid=in_grid,
        )

        # --- Teacher (VJEPA2) ---
        self.use_vjepa_loss = use_vjepa_loss
        self.teacher_model = None

        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)

        # align cfg
        self.align_common_dim = align_common_dim
        self.align_num_prototypes = align_num_prototypes
        self.align_kmeans_iters = align_kmeans_iters
        self.align_kmeans_temp = align_kmeans_temp
        self.align_gram_weight = align_gram_weight
        self.align_pca_weight = align_pca_weight
        self.align_pca_rank = align_pca_rank

        # build teacher/aligner
        if self.use_vjepa_loss:
            self._init_vjepa2_teacher(student_dim=token_size)

    def _init_vjepa2_teacher(self, student_dim: int):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return

        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
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
        self.prior_model = None 
        # aligner
        self.aligner = VJepaAlignerV3(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            student_grid=self.latent_grid,
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
        x: [B, 3, T, H, W], 期望是 0..1 的浮点
        输出：x_teacher [B, 3, T', Ht, Wt]，已做 resize/crop + imagenet normalize
        """
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.float()#B,3,T,H,W
        # 采样到 teacher 需要的帧数（例如64）
        #x = self._sample_or_pad_frames(x, self.vjepa2_num_frames)  # [B,3,T',H,W]

        # transform 期望输入为 [T, C, H, W]（单样本）
        # 批处理：循环做（可接受；若想更快可把 transform 改为纯 tensor op）
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

    def encode(self, data, **kwargs):
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        x_q, out_dict = self.encode(x)
        pred_x = self.decode(x_q)

        return_dict = {"pred_frames": pred_x.contiguous()}

        if self.training and self.use_vjepa_loss and (self.teacher_model is not None):
            # teacher to device
            if next(self.teacher_model.parameters()).device != x.device:
                self.teacher_model.to(x.device)

            with torch.no_grad():
                t_input = self._preprocess_for_vjepa2(x)        # [B,3,T',Ht,Wt]
                t_feats = self.teacher_model(t_input)           # [B,Nt,Dt] (可能含CLS)
                #print(f"Teacher output tokens: {t_feats.shape}")
                t_feats = t_feats.float()

            # teacher grid shape 推断（与 ViT patch/tubelet 对齐）
            # 注意：这里用 vjepa2_num_frames / tubelet_size；img_size / patch_size
            tt = self.vjepa2_num_frames // self.vjepa2_tubelet_size
            ht = self.vjepa2_img_size // self.vjepa2_patch_size
            wt = self.vjepa2_img_size // self.vjepa2_patch_size

            align_loss, align_dict = self.aligner(x_q.float(), t_feats, (tt, ht, wt))
            return_dict["align_loss"] =0.5* align_loss
            #return_dict.update({f"align_{k}": v for k, v in align_dict.items()})

        return return_dict
    



@register("autoencoder_cnnvit_softalign_gram_vic_vjepa2")
class AutoEncoder1(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=128,

        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vitl.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=256,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        # --- Alignment config ---
        align_common_dim=256,
        align_num_prototypes=256,
        align_kmeans_iters=5,
        align_kmeans_temp=0.2,
        align_gram_weight=1.0,
        align_pca_weight=0.2,
        align_pca_rank=32,

        **kwargs,
    ):
        super().__init__()

        self.latent_grid = (4, 16, 16)  # student tokens grid: t,h,w (4*16*16=1024)
        in_grid = [16, 128, 128]
        token_size = 6  # student_dim

        # --- Student ---
        self.encoder = Encoder(
            model_size="base_thin",
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=num_latent_tokens,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])
        self.decoder = Decoder(
            model_size="base_thin",
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=num_latent_tokens,
            out_grid=in_grid,
        )

        # --- Teacher (VJEPA2) ---
        self.use_vjepa_loss = use_vjepa_loss
        self.teacher_model = None

        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)

        # align cfg
        self.align_common_dim = align_common_dim
        self.align_num_prototypes = align_num_prototypes
        self.align_kmeans_iters = align_kmeans_iters
        self.align_kmeans_temp = align_kmeans_temp
        self.align_gram_weight = align_gram_weight
        self.align_pca_weight = align_pca_weight
        self.align_pca_rank = align_pca_rank

        # build teacher/aligner
        if self.use_vjepa_loss:
            self._init_vjepa2_teacher(student_dim=token_size)

    def _init_vjepa2_teacher(self, student_dim: int):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            return

        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        # vit giant rope variant used in demo
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
        self.prior_model = None 
        # aligner
        self.aligner = VJepaAlignerV2(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            student_grid=self.latent_grid,
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
        x: [B, 3, T, H, W], 期望是 0..1 的浮点
        输出：x_teacher [B, 3, T', Ht, Wt]，已做 resize/crop + imagenet normalize
        """
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.float()#B,3,T,H,W
        # 采样到 teacher 需要的帧数（例如64）
        #x = self._sample_or_pad_frames(x, self.vjepa2_num_frames)  # [B,3,T',H,W]

        # transform 期望输入为 [T, C, H, W]（单样本）
        # 批处理：循环做（可接受；若想更快可把 transform 改为纯 tensor op）
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

    def encode(self, data, **kwargs):
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        x: [B, 3, 16, 128, 128]
        """
        x_q, out_dict = self.encode(x)
        pred_x = self.decode(x_q)

        return_dict = {"pred_frames": pred_x.contiguous()}

        if self.training and self.use_vjepa_loss and (self.teacher_model is not None):
            # teacher to device
            if next(self.teacher_model.parameters()).device != x.device:
                self.teacher_model.to(x.device)

            with torch.no_grad():
                t_input = self._preprocess_for_vjepa2(x)        # [B,3,T',Ht,Wt]
                t_feats = self.teacher_model(t_input)           # [B,Nt,Dt] (可能含CLS)
                #print(f"Teacher output tokens: {t_feats.shape}")
                t_feats = t_feats.float()

            # teacher grid shape 推断（与 ViT patch/tubelet 对齐）
            # 注意：这里用 vjepa2_num_frames / tubelet_size；img_size / patch_size
            tt = self.vjepa2_num_frames // self.vjepa2_tubelet_size
            ht = self.vjepa2_img_size // self.vjepa2_patch_size
            wt = self.vjepa2_img_size // self.vjepa2_patch_size

            align_loss, align_dict = self.aligner(x_q.float(), t_feats, (tt, ht, wt))
            return_dict["align_loss"] =0.5* align_loss
            #return_dict.update({f"align_{k}": v for k, v in align_dict.items()})

        return return_dict