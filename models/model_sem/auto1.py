
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
    #short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(128, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


@register("autoencoder_vfm")
class AutoEncoder(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=512,
        input_size=128,

        # --- VJEPA2 Teacher (PyTorch-only) ---
        use_vjepa_loss=True,
        vjepa2_encoder_ckpt="/data2/zhxie/myproject/LARP/vjepackpt/vitl.pt",  # 本地 .pth/.tar，包含 ckpt["encoder"]
        vjepa2_img_size=128,                    # 与所选 VJEPA2 权重对应的 crop_size（如 384/256）
        vjepa2_num_frames=16,                   # VJEPA2 encoder 构建时 num_frames
        vjepa2_sample_strategy="repeat",        # "repeat" | "uniform"
        vjepa2_tubelet_size=2,                  # VJEPA2 常见为2（demo里采样到64帧=128/2）
        vjepa2_patch_size=16,                   # ViT-G 常见 patch=16（用于 grid 计算）
        vjepa2_use_bf16=False,                  # teacher 用 bf16 可省显存

        **kwargs,
    ):
        super().__init__()

        self.vfm_grid = (8, 8, 8)  # student tokens grid: t,h,w (4*16*16=1024)
        token_size = 6  # student_dim
        # --- Student ---
        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='small',
            in_channels=1024,
            out_channels=token_size,
            in_tokens=512,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])

        self.tokenizer_decoder = TokenizerDecoder1D(
            model_size='small',
            in_channels=token_size,
            in_tokens=num_latent_tokens,
            out_tokens=512,
            out_grid=self.vfm_grid,
        )
        decoder_1d_width = self.tokenizer_decoder.output_dim
        
        # Video Decoder: width → RGB video
        self.video_decoder = VideoDecoder(
            model_size='small',
            in_channels=decoder_1d_width,
            out_channels=3,
            num_tokens=512,
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
        self.aligner= nn.Linear(decoder_1d_width, 1024) 
        self.dicklinear= nn.Linear(1024, decoder_1d_width)
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
        return_dict["align_loss"] = 0.5 * align_loss

        return return_dict
    



