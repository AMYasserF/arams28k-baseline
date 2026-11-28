"""Unified Script-Aware Hybrid CTC/Attention HTR Architecture.

Combines:
  1. Asymmetric-stride ResNet-34 backbone (vertical downsampling only in layer2..4 -> H/32=2, W/4).
  2. Masked global mean+std Style/Script FiLM conditioning + optional Aux Script classifier.
  3. Multi-scale Stage 2 + Stage 4 fusion + Learned Vertical Attention Pooling (H=2 -> 1).
  4. 2-layer BiLSTM sequence encoder projecting cleanly to 768 dims.
  5. Dual heads:
       - Auxiliary CTC head (38 classes: <blank>=0 + 37 NFC Arabic chars).
       - Pretrained HATFormer TrOCRForCausalLM decoder (d_model=1024, cross_attention_hidden_size=768,
         vocab=50,265) with decoder context dropout during training.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrOCRConfig, TrOCRForCausalLM

try:
    from transformers import DynamicCache, EncoderDecoderCache
except ImportError:  # pragma: no cover
    DynamicCache = None
    EncoderDecoderCache = None

try:
    from torchvision.models import ResNet34_Weights, resnet34 as _tv_resnet34
except ImportError:  # pragma: no cover
    ResNet34_Weights = None
    _tv_resnet34 = None


TROCR_BOS_ID: int = 0
TROCR_PAD_ID: int = 1
TROCR_EOS_ID: int = 2
TROCR_UNK_ID: int = 3
TROCR_MASK_ID: int = 4
TROCR_VOCAB_SIZE: int = 50265
HATFORMER_ENC_DIM: int = 768


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int = TROCR_PAD_ID,
    decoder_start_token_id: int = TROCR_BOS_ID,
) -> torch.Tensor:
    """Shift target token IDs right by 1 position, prepending decoder_start_token_id and masking -100 -> pad."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def default_hatformer_decoder_config() -> TrOCRConfig:
    """Exact verified TrOCRForCausalLM decoder configuration of Archatext/hatformer-arams28k & muharaf."""
    return TrOCRConfig(
        vocab_size=TROCR_VOCAB_SIZE,
        d_model=1024,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        classifier_dropout=0.0,
        cross_attention_hidden_size=HATFORMER_ENC_DIM,
        max_position_embeddings=512,
        layernorm_embedding=True,
        scale_embedding=False,
        use_learned_position_embeddings=True,
        tie_word_embeddings=True,
        is_decoder=True,
        add_cross_attention=True,
        bos_token_id=TROCR_BOS_ID,
        pad_token_id=TROCR_PAD_ID,
        eos_token_id=TROCR_EOS_ID,
        decoder_start_token_id=TROCR_BOS_ID,
        use_cache=False,
    )


def _resolve_muharaf_local_dir() -> Path | None:
    """Locate hatformer-muharaf checkpoint across standard repo and Kaggle paths."""
    repo_root = Path(__file__).resolve().parents[3]
    direct_candidates = [
        repo_root / "checkpoints" / "hatformer" / "original" / "hatformer-muharaf",
        repo_root / "checkpoints" / "hatformer" / "hatformer-muharaf",
        Path("/kaggle/temp/hatformer/checkpoints/hatformer/original/hatformer-muharaf"),
    ]
    for c in direct_candidates:
        if c.is_dir() and (c / "config.json").is_file():
            return c

    for search_root in (repo_root / "checkpoints", Path("/kaggle/input"), Path("/kaggle/temp")):
        if not search_root.is_dir():
            continue
        for dp, _, fn in os.walk(search_root):
            if "config.json" in fn and "muharaf" in dp.lower():
                has_weights = any(f.endswith((".safetensors", ".bin")) for f in fn)
                if has_weights:
                    return Path(dp)
    return None


def _load_state_dict_from_dir(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    st_path = ckpt_dir / "model.safetensors"
    if st_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    idx_st = ckpt_dir / "model.safetensors.index.json"
    if idx_st.is_file():
        from safetensors.torch import load_file
        weight_map = json.loads(idx_st.read_text(encoding="utf-8"))["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(str(ckpt_dir / shard), device="cpu"))
        return sd
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {ckpt_dir}")


def load_trocr_decoder(
    ckpt_or_hub_id: str = "hatformer-muharaf",
    torch_dtype: torch.dtype = torch.float32,
) -> TrOCRForCausalLM:
    """Load standalone HATFormer TrOCRForCausalLM decoder weights cleanly without ViT/BEiT.

    Defaults to `hatformer-muharaf` (exact pre-AraMS initialization used by baseline HATFormer)
    and supports any local VisionEncoderDecoderModel directory or Hugging Face repo ID
    (e.g., `Archatext/hatformer-arams28k`). Forces `decoder_start_token_id=0` (<s>).
    """
    cand = Path(ckpt_or_hub_id)
    if cand.is_dir():
        raw_cfg = json.loads((cand / "config.json").read_text(encoding="utf-8"))
        raw_sd = _load_state_dict_from_dir(cand)
    elif ckpt_or_hub_id == "hatformer-muharaf":
        local_dir = _resolve_muharaf_local_dir()
        if local_dir is None:
            raise FileNotFoundError(
                "Could not locate `hatformer-muharaf` locally under checkpoints/ or /kaggle/input/. "
                "Download/unzip https://zenodo.org/records/14165756/files/HATFormer.zip?download=1 or pass "
                "an explicit local directory / Hugging Face repo ID to load_trocr_decoder()."
            )
        raw_cfg = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
        raw_sd = _load_state_dict_from_dir(local_dir)
    else:
        from huggingface_hub import hf_hub_download

        cfg_file = hf_hub_download(repo_id=ckpt_or_hub_id, filename="config.json")
        raw_cfg = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
        weight_file = None
        for fname in ("model.safetensors", "pytorch_model.bin"):
            try:
                weight_file = hf_hub_download(repo_id=ckpt_or_hub_id, filename=fname)
                break
            except Exception:
                continue
        if weight_file is None:
            raise FileNotFoundError(f"Could not download safetensors/bin weights from HF repo {ckpt_or_hub_id}")
        if weight_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(weight_file, device="cpu")
        else:
            raw_sd = torch.load(weight_file, map_location="cpu", weights_only=True)

    dec_dict = dict(raw_cfg["decoder"]) if "decoder" in raw_cfg else dict(raw_cfg)
    dec_dict.update({
        "is_decoder": True,
        "add_cross_attention": True,
        "bos_token_id": TROCR_BOS_ID,
        "pad_token_id": TROCR_PAD_ID,
        "eos_token_id": TROCR_EOS_ID,
        "decoder_start_token_id": TROCR_BOS_ID,  # CRITICAL: nested config defaults to 2 (</s>)
        "use_cache": False,
    })

    dec_cfg = TrOCRConfig(**dec_dict)
    decoder = TrOCRForCausalLM(dec_cfg)

    has_prefix = any(k.startswith("decoder.") for k in raw_sd)
    dec_sd = {
        (k[len("decoder."):] if k.startswith("decoder.") else k): v
        for k, v in raw_sd.items()
        if (k.startswith("decoder.") if has_prefix else True)
    }

    missing, unexpected = decoder.load_state_dict(dec_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading TrOCRForCausalLM: {unexpected}")

    # VisionEncoderDecoderModel safetensors omits tied output_projection.weight; re-tie cleanly
    decoder.tie_weights()
    allowed_missing = {"output_projection.weight"} if dec_cfg.tie_word_embeddings else set()
    real_missing = set(missing) - allowed_missing
    if real_missing:
        raise RuntimeError(f"Missing keys when loading TrOCRForCausalLM: {sorted(real_missing)}")
    if dec_cfg.tie_word_embeddings:
        assert decoder.output_projection.weight is decoder.model.decoder.embed_tokens.weight, (
            "Failed to tie output_projection.weight to embed_tokens.weight"
        )

    decoder = decoder.to(dtype=torch_dtype)
    decoder.config.decoder_start_token_id = TROCR_BOS_ID
    decoder.config.bos_token_id = TROCR_BOS_ID
    decoder.config.pad_token_id = TROCR_PAD_ID
    decoder.config.eos_token_id = TROCR_EOS_ID
    gc = decoder.generation_config
    gc.decoder_start_token_id = TROCR_BOS_ID
    gc.bos_token_id = TROCR_BOS_ID
    gc.pad_token_id = TROCR_PAD_ID
    gc.eos_token_id = TROCR_EOS_ID
    return decoder


# ─────────────────────────────────────────────────────────────────────────────
# Torchvision-Compatible Asymmetric-Stride ResNet-34 Backbone
# ─────────────────────────────────────────────────────────────────────────────

class _CompatBasicBlock(nn.Module):
    """Fallback ResNet BasicBlock matching torchvision state_dict keys exactly."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _CompatResNet34(nn.Module):
    """Minimal fallback ResNet-34 matching torchvision attribute & state_dict layout."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_CompatBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_CompatBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)


def build_asymmetric_resnet34(pretrained: bool = True) -> nn.Module:
    """Construct ResNet-34 with vertical-only strides (2, 1) in layer2..4 -> H/32, W/4."""
    if _tv_resnet34 is not None:
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = _tv_resnet34(weights=weights)
        # Drop classification head unused by HTR encoder
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
    else:
        backbone = _CompatResNet34()

    for layer_name in ("layer2", "layer3", "layer4"):
        block0 = getattr(backbone, layer_name)[0]
        block0.conv1.stride = (2, 1)
        if block0.downsample is not None:
            block0.downsample[0].stride = (2, 1)

    return backbone


# ─────────────────────────────────────────────────────────────────────────────
# Style / Script Conditioning (FiLM)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mean_std_pool(feat: torch.Tensor, mask_1d: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute masked [mean, std] across spatial (H, T) excluding right-padded columns."""
    bsz, ch, h, t_len = feat.shape
    m = mask_1d.to(dtype=feat.dtype).view(bsz, 1, 1, t_len)
    valid_elems = (m.sum(dim=(2, 3), keepdim=False) * float(h)).clamp(min=1.0)  # (B, 1)

    masked_feat = feat * m
    mean = masked_feat.sum(dim=(2, 3)) / valid_elems  # (B, C)
    diff = (feat - mean.unsqueeze(-1).unsqueeze(-1)) * m
    var = (diff * diff).sum(dim=(2, 3)) / valid_elems
    std = torch.sqrt(var.clamp(min=0.0) + eps)
    return torch.cat([mean, std], dim=-1)  # (B, 2*C)


class StyleFiLMConditioner(nn.Module):
    """Extract masked Stage 2 + Stage 4 style vector, apply zero-init FiLM, & predict script."""

    def __init__(
        self,
        s2_dim: int = 128,
        s4_dim: int = 512,
        feat_dim: int = 512,
        style_dim: int = 256,
        num_scripts: int = 4,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = 2 * s2_dim + 2 * s4_dim  # 1280
        self.pool_drop = nn.Dropout(dropout_p)
        self.style_proj = nn.Sequential(
            nn.Linear(in_dim, style_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, 2 * feat_dim),
        )
        # Zero-init final FiLM layer so model begins strictly as identity: (1 + 0) * x + 0 = x
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        self.script_head = nn.Linear(style_dim, num_scripts)

    def forward(
        self,
        s2: torch.Tensor,
        s4: torch.Tensor,
        feat: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=feat.device.type, enabled=False):
            stats_s2 = masked_mean_std_pool(s2.float(), encoder_attention_mask)
            stats_s4 = masked_mean_std_pool(s4.float(), encoder_attention_mask)
            s_raw = self.pool_drop(torch.cat([stats_s2, stats_s4], dim=-1))
            style_vec = self.style_proj(s_raw)
            gamma, beta = self.film_mlp(style_vec).chunk(2, dim=-1)
            modulated = (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) * feat.float() + beta.unsqueeze(-1).unsqueeze(-1)
            script_logits = self.script_head(style_vec)
        return modulated.to(dtype=feat.dtype), script_logits


# ─────────────────────────────────────────────────────────────────────────────
# Full Unified Hybrid CTC / Attention Model
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedHTRModel(nn.Module):
    """Unified variable-width CNN + FiLM + BiLSTM + Aux-CTC + TrOCR Decoder HTR system."""

    def __init__(
        self,
        decoder: TrOCRForCausalLM | None = None,
        decoder_config: TrOCRConfig | None = None,
        ctc_vocab_size: int = 38,
        num_scripts: int = 4,
        use_multiscale: bool = True,
        use_film: bool = True,
        use_ctc: bool = True,
        use_script_loss: bool = True,
        film_dropout: float = 0.2,
        spec_aug_prob: float = 0.0,
        spec_aug_num_masks: int = 2,
        spec_aug_max_ratio: float = 0.08,
        enc_feat_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        context_dropout: float = 0.10,
        lambda_ctc: float = 0.30,
        alpha_script: float = 0.05,
        pretrained_backbone: bool = True,
        lstm_hidden_per_dir: int = 384,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ctc_vocab_size = int(ctc_vocab_size)
        self.num_scripts = int(num_scripts)
        self.use_multiscale = bool(use_multiscale)
        self.use_film = bool(use_film)
        self.use_ctc = bool(use_ctc)
        self.use_script_loss = bool(use_script_loss) and self.use_film
        self.film_dropout = float(film_dropout)
        self.spec_aug_prob = float(spec_aug_prob)
        self.spec_aug_num_masks = int(spec_aug_num_masks)
        self.spec_aug_max_ratio = float(spec_aug_max_ratio)
        self.enc_feat_dropout = float(enc_feat_dropout)
        self.label_smoothing = float(label_smoothing)
        self.context_dropout = float(context_dropout)
        self.lambda_ctc = float(lambda_ctc)
        self.alpha_script = float(alpha_script)

        # 1. Asymmetric-stride ResNet-34
        self.backbone = build_asymmetric_resnet34(pretrained=pretrained_backbone)

        # 2. Multi-scale fusion (Stage 2 H=8 -> H=2 + Stage 4 H=2 -> 512 ch)
        if self.use_multiscale:
            self.s2_down = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))
            self.ms_fuse = nn.Sequential(
                nn.Conv2d(128 + 512, 512, kernel_size=1, bias=False),
                nn.GroupNorm(32, 512),
                nn.GELU(),
            )

        # 3. Masked Style/Script FiLM Conditioning
        if self.use_film:
            self.film = StyleFiLMConditioner(
                s2_dim=128,
                s4_dim=512,
                feat_dim=512,
                style_dim=256,
                num_scripts=self.num_scripts,
                dropout_p=self.film_dropout,
            )

        # 4. Learned Vertical Attention Pooling (H=2 -> 1) + Pre-BiLSTM Regularization
        self.vert_attn = nn.Conv2d(512, 1, kernel_size=1)
        self.enc_feat_drop = nn.Dropout(self.enc_feat_dropout)

        # 5. 1D Sequence BiLSTM Encoder + 768-dim residual projection
        self.seq_lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_per_dir,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        seq_dim = 2 * lstm_hidden_per_dir
        assert seq_dim == HATFORMER_ENC_DIM, f"BiLSTM output must be {HATFORMER_ENC_DIM}, got {seq_dim}"
        self.enc_proj = nn.Linear(seq_dim, HATFORMER_ENC_DIM)
        self.enc_norm = nn.LayerNorm(HATFORMER_ENC_DIM)

        # 6. Auxiliary CTC Head
        if self.use_ctc:
            self.ctc_head = nn.Linear(HATFORMER_ENC_DIM, self.ctc_vocab_size)

        # 7. Standalone TrOCRForCausalLM Decoder
        if decoder is not None:
            self.decoder = decoder
        else:
            dcfg = decoder_config or default_hatformer_decoder_config()
            self.decoder = TrOCRForCausalLM(dcfg)
        self._enforce_decoder_special_ids()

    def _enforce_decoder_special_ids(self) -> None:
        self.decoder.config.is_decoder = True
        self.decoder.config.add_cross_attention = True
        self.decoder.config.bos_token_id = TROCR_BOS_ID
        self.decoder.config.pad_token_id = TROCR_PAD_ID
        self.decoder.config.eos_token_id = TROCR_EOS_ID
        self.decoder.config.decoder_start_token_id = TROCR_BOS_ID
        gc = self.decoder.generation_config
        gc.bos_token_id = TROCR_BOS_ID
        gc.pad_token_id = TROCR_PAD_ID
        gc.eos_token_id = TROCR_EOS_ID
        gc.decoder_start_token_id = TROCR_BOS_ID

    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_ckpt_or_hub_id: str = "hatformer-muharaf",
        **kwargs: Any,
    ) -> "UnifiedHTRModel":
        decoder = load_trocr_decoder(decoder_ckpt_or_hub_id)
        return cls(decoder=decoder, **kwargs)

    def set_decoder_frozen(self, frozen: bool) -> None:
        """Freeze (Stage 1) or unfreeze (Stage 2) all TrOCR decoder parameters."""
        for p in self.decoder.parameters():
            p.requires_grad = not frozen

    @staticmethod
    def _pad_pixel_values_to_stride4(pixel_values: torch.Tensor) -> torch.Tensor:
        """Right-pad image width with 0.0 to a multiple of 4 so CNN output width is exact W // 4."""
        w = int(pixel_values.shape[-1])
        rem = w % 4
        if rem != 0:
            pixel_values = F.pad(pixel_values, (0, 4 - rem), mode="constant", value=0.0)
        return pixel_values

    @staticmethod
    def _align_encoder_mask_and_lengths(
        encoder_attention_mask: torch.Tensor | None,
        enc_lengths: torch.Tensor | None,
        t_max: int,
        bsz: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align 2D encoder_attention_mask and 1D enc_lengths strictly to spatial width t_max."""
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.to(device=device, dtype=torch.long)
            if mask.ndim != 2 or mask.shape[0] != bsz:
                raise ValueError(
                    f"encoder_attention_mask must have shape ({bsz}, T), got {tuple(mask.shape)}"
                )
            m_len = int(mask.shape[1])
            if m_len < t_max:
                mask = F.pad(mask, (0, t_max - m_len), mode="constant", value=0)
            elif m_len > t_max:
                mask = mask[:, :t_max]
        elif enc_lengths is not None:
            lens_tmp = enc_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=t_max)
            pos = torch.arange(t_max, device=device, dtype=torch.long).unsqueeze(0)
            mask = (pos < lens_tmp.unsqueeze(1)).long()
        else:
            mask = torch.ones((bsz, t_max), device=device, dtype=torch.long)

        if enc_lengths is not None:
            lens = enc_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=t_max)
        else:
            lens = mask.sum(dim=1).clamp(min=1, max=t_max).long()
        return mask, lens

    def _ensure_cross_attn_mask(
        self,
        enc_features: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Align and validate encoder_attention_mask directly against encoder_hidden_states.shape[1]."""
        bsz, src_len = int(enc_features.shape[0]), int(enc_features.shape[1])
        mask, _ = self._align_encoder_mask_and_lengths(
            encoder_attention_mask=encoder_attention_mask,
            enc_lengths=None,
            t_max=src_len,
            bsz=bsz,
            device=enc_features.device,
        )
        if mask.shape != (bsz, src_len):
            raise ValueError(
                f"Cross-attention mask shape mismatch: expected {(bsz, src_len)}, got {tuple(mask.shape)}"
            )
        return mask

    def _decoder_uses_encoder_decoder_cache(self) -> bool:
        """Detect transformers >= 5.x Cache-based TrOCRDecoder where EncoderDecoderCache is required."""
        if DynamicCache is None or EncoderDecoderCache is None:
            return False
        supports_fn = getattr(self.decoder, "_supports_default_dynamic_cache", None)
        if not (callable(supports_fn) and bool(supports_fn())):
            return False
        try:
            layers = self.decoder.model.decoder.layers
            if len(layers) == 0:
                return False
            enc_attn = getattr(layers[0], "encoder_attn", None)
            if enc_attn is None:
                return False
            return "past_key_values" in inspect.signature(enc_attn.forward).parameters
        except Exception:
            return False

    def _build_decoder_past_key_values(self) -> Any | None:
        """Build EncoderDecoderCache so self-attn and cross-attn do not share a single DynamicCache."""
        if not self._decoder_uses_encoder_decoder_cache():
            return None
        dc_params = inspect.signature(DynamicCache.__init__).parameters
        dc_kwargs = {"config": self.decoder.config} if "config" in dc_params else {}
        return EncoderDecoderCache(DynamicCache(**dc_kwargs), DynamicCache(**dc_kwargs))

    def encode(
        self,
        pixel_values: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        enc_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run CNN + optional MultiScale/FiLM + Vertical Attn Pool + BiLSTM -> (B, T, 768)."""
        pixel_values = self._pad_pixel_values_to_stride4(pixel_values)
        x = self.backbone.conv1(pixel_values)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        s1 = self.backbone.layer1(x)
        s2 = self.backbone.layer2(s1)  # (B, 128, 8, W/4)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)  # (B, 512, 2, W/4)

        bsz, _, _, t_max = s4.shape
        aligned_mask, aligned_lengths = self._align_encoder_mask_and_lengths(
            encoder_attention_mask=encoder_attention_mask,
            enc_lengths=enc_lengths,
            t_max=t_max,
            bsz=bsz,
            device=s4.device,
        )
        col_mask = aligned_mask.to(dtype=s4.dtype).view(bsz, 1, 1, t_max)
        s2_masked = s2 * col_mask
        s4_masked = s4 * col_mask

        if self.use_multiscale:
            s2_ds = self.s2_down(s2_masked)  # (B, 128, 2, W/4)
            feat = self.ms_fuse(torch.cat([s2_ds, s4_masked], dim=1))  # (B, 512, 2, W/4)
        else:
            feat = s4_masked

        script_logits: torch.Tensor | None = None
        if self.use_film:
            feat, script_logits = self.film(s2_masked, s4_masked, feat, aligned_mask)
            feat = feat * col_mask

        # Vertical attention pooling across H=2 -> 1
        attn_weights = F.softmax(self.vert_attn(feat), dim=2)  # (B, 1, 2, T)
        seq_in = (feat * attn_weights).sum(dim=2).transpose(1, 2).contiguous()  # (B, T, 512)
        seq_in = self._apply_spec_augment(seq_in, aligned_lengths)
        if self.enc_feat_dropout > 0.0:
            seq_in = self.enc_feat_drop(seq_in)

        # Pack padded sequence for BiLSTM so padding never contaminates recurrent states
        safe_lens = aligned_lengths.detach().cpu().clamp(min=1, max=t_max)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_in, safe_lens, batch_first=True, enforce_sorted=False
        )
        # Enforce FP32 inside RNN if autocast causes CUDNN issues or run cleanly
        packed_out, _ = self.seq_lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=t_max
        )
        enc_features = self.enc_norm(lstm_out + self.enc_proj(lstm_out))
        enc_features = enc_features * aligned_mask.unsqueeze(-1).to(dtype=enc_features.dtype)

        ctc_logits = self.ctc_head(enc_features) if self.use_ctc else None
        return {
            "enc_features": enc_features,
            "encoder_attention_mask": aligned_mask,
            "enc_lengths": aligned_lengths,
            "ctc_logits": ctc_logits,
            "script_logits": script_logits,
        }

    def _apply_spec_augment(self, seq_in: torch.Tensor, enc_lengths: torch.Tensor) -> torch.Tensor:
        """Mask random contiguous valid encoder spans per sample during training only."""
        if (
            (not self.training)
            or self.spec_aug_prob <= 0.0
            or self.spec_aug_num_masks <= 0
            or self.spec_aug_max_ratio <= 0.0
        ):
            return seq_in

        bsz, t_max, _ = seq_in.shape
        keep_mask = torch.ones((bsz, t_max, 1), device=seq_in.device, dtype=seq_in.dtype)
        for i in range(bsz):
            if float(torch.rand((), device="cpu").item()) >= self.spec_aug_prob:
                continue
            valid_len = max(1, min(t_max, int(enc_lengths[i].item())))
            if valid_len <= 1:
                continue
            max_span = min(max(1, int(valid_len * self.spec_aug_max_ratio)), valid_len - 1)
            for _ in range(self.spec_aug_num_masks):
                span = int(torch.randint(1, max_span + 1, ()).item())
                start_max = max(0, valid_len - span)
                start = int(torch.randint(0, start_max + 1, ()).item())
                keep_mask[i, start : start + span, 0] = 0.0
        return seq_in * keep_mask

    def _apply_context_dropout(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.context_dropout <= 0.0:
            return decoder_input_ids
        pad_id = TROCR_PAD_ID
        bos_id = TROCR_BOS_ID
        eos_id = TROCR_EOS_ID
        special = (decoder_input_ids == bos_id) | (decoder_input_ids == pad_id) | (decoder_input_ids == eos_id)
        drop_mask = (~special) & (
            torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < self.context_dropout
        )
        return decoder_input_ids.masked_fill(drop_mask, TROCR_MASK_ID)

    def _compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        ctc_targets: torch.Tensor,
        enc_lengths: torch.Tensor,
        ctc_target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        valid = (ctc_target_lengths > 0) & (enc_lengths >= ctc_target_lengths)
        if not bool(valid.any()):
            return ctc_logits.float().sum() * 0.0
        with torch.amp.autocast(device_type=ctc_logits.device.type, enabled=False):
            log_probs = F.log_softmax(ctc_logits[valid].float(), dim=-1).transpose(0, 1)  # (T, B_v, V)
            loss = F.ctc_loss(
                log_probs,
                ctc_targets[valid],
                enc_lengths[valid].clamp(min=1),
                ctc_target_lengths[valid],
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        enc_lengths: torch.Tensor,
        decoder_labels: torch.Tensor,
        ctc_targets: torch.Tensor | None = None,
        ctc_target_lengths: torch.Tensor | None = None,
        script_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute joint CE + Aux-CTC + Aux-Script loss."""
        enc = self.encode(pixel_values, encoder_attention_mask, enc_lengths)
        enc_features = enc["enc_features"]
        aligned_mask = self._ensure_cross_attn_mask(enc_features, enc["encoder_attention_mask"])
        aligned_lengths = enc["enc_lengths"]
        ctc_logits = enc["ctc_logits"]
        script_logits = enc["script_logits"]

        dec_input_ids = shift_tokens_right(
            decoder_labels, pad_token_id=TROCR_PAD_ID, decoder_start_token_id=TROCR_BOS_ID
        )
        dec_input_ids = self._apply_context_dropout(dec_input_ids)
        dec_attn_mask = (dec_input_ids != TROCR_PAD_ID).long()

        use_smoothing = self.label_smoothing > 0.0
        dec_out = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=enc_features,
            encoder_attention_mask=aligned_mask,
            labels=(None if use_smoothing else decoder_labels),
            use_cache=False,
        )
        if use_smoothing:
            logits = dec_out.logits.float()
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_ce = dec_out.loss

        zero = loss_ce.new_zeros(())
        if self.use_ctc and ctc_logits is not None and ctc_targets is not None and ctc_target_lengths is not None:
            loss_ctc = self._compute_ctc_loss(ctc_logits, ctc_targets, aligned_lengths, ctc_target_lengths)
            eff_lambda = self.lambda_ctc
        else:
            loss_ctc = zero
            eff_lambda = 0.0

        if self.use_script_loss and script_logits is not None and script_ids is not None:
            loss_script = F.cross_entropy(script_logits, script_ids)
            eff_alpha = self.alpha_script
        else:
            loss_script = zero
            eff_alpha = 0.0

        total_loss = (1.0 - eff_lambda) * loss_ce + eff_lambda * loss_ctc + eff_alpha * loss_script
        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_ctc": loss_ctc,
            "loss_script": loss_script,
            "ctc_logits": ctc_logits,
            "script_logits": script_logits,
            "enc_features": enc_features,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        enc_lengths: torch.Tensor,
        tokenizer: Any,
        ctc_vocab: Any | None = None,
        num_beams: int = 3,
        max_new_tokens: int = 64,
        return_ctc: bool = False,
        **gen_kwargs: Any,
    ) -> list[str] | tuple[list[str], list[str]]:
        """Generate decoded Arabic strings (and optional CTC greedy strings)."""
        was_training = self.training
        self.eval()
        enc = self.encode(pixel_values, encoder_attention_mask, enc_lengths)
        enc_features = enc["enc_features"]
        aligned_mask = self._ensure_cross_attn_mask(enc_features, enc["encoder_attention_mask"])
        aligned_lengths = enc["enc_lengths"]

        kw: dict[str, Any] = {
            "encoder_hidden_states": enc_features,
            "encoder_attention_mask": aligned_mask,
            "decoder_start_token_id": TROCR_BOS_ID,
            "bos_token_id": TROCR_BOS_ID,
            "pad_token_id": TROCR_PAD_ID,
            "eos_token_id": TROCR_EOS_ID,
            "num_beams": int(num_beams),
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
        }
        if int(num_beams) > 1 and "length_penalty" not in gen_kwargs:
            kw["length_penalty"] = 0.0
        kw.update(gen_kwargs)
        if bool(kw.get("use_cache", True)) and "past_key_values" not in kw:
            past_kv = self._build_decoder_past_key_values()
            if past_kv is not None:
                kw["past_key_values"] = past_kv

        out_ids = self.decoder.generate(**kw)
        dec_texts = tokenizer.batch_decode(out_ids.tolist(), skip_special_tokens=True)

        if not return_ctc:
            if was_training:
                self.train()
            return dec_texts

        ctc_texts: list[str] = []
        ctc_logits = enc["ctc_logits"]
        if ctc_logits is not None and ctc_vocab is not None:
            pred_ids = ctc_logits.argmax(dim=-1).cpu()
            for i in range(pred_ids.shape[0]):
                t_i = int(aligned_lengths[i].item())
                ctc_texts.append(ctc_vocab.decode_greedy(pred_ids[i, :t_i].tolist()))
        else:
            ctc_texts = ["" for _ in dec_texts]

        if was_training:
            self.train()
        return dec_texts, ctc_texts
