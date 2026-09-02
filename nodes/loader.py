"""Fish S2-Pro checkpoint discovery and native ComfyUI model loading."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

import comfy.model_management as model_management
from comfy.model_patcher import CoreModelPatcher, ModelPatcher

logger = logging.getLogger("FishAudioS2")

MODELS_FOLDER_NAME = "fishs2"
HF_DEFAULT_MODEL_NAME = "s2-pro"
AUTO_DOWNLOAD_SUFFIX = " (auto download)"


def _get_models_base() -> Path:
    import folder_paths

    base = Path(folder_paths.models_dir) / MODELS_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _register_folder() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, str(_get_models_base()))


def get_model_names() -> list[str]:
    base = _get_models_base()
    names = sorted(path.name for path in base.iterdir() if path.is_dir())
    if HF_DEFAULT_MODEL_NAME not in names:
        names.insert(0, f"{HF_DEFAULT_MODEL_NAME}{AUTO_DOWNLOAD_SUFFIX}")
    return names or [f"{HF_DEFAULT_MODEL_NAME}{AUTO_DOWNLOAD_SUFFIX}"]


def _strip_auto_download_suffix(model_name: str) -> str:
    return model_name.removesuffix(AUTO_DOWNLOAD_SUFFIX)


def _download_official_checkpoint(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="fishaudio/s2-pro", local_dir=str(destination))


def resolve_model_path(model_name: str) -> Path:
    requested_download = model_name.endswith(AUTO_DOWNLOAD_SUFFIX)
    model_name = _strip_auto_download_suffix(model_name)
    if model_name != HF_DEFAULT_MODEL_NAME:
        raise RuntimeError(
            f"'{model_name}' is not supported by the native Fish S2 integration. "
            "This release supports only the official s2-pro BF16 checkpoint; "
            "FP8 and BitsAndBytes support requires ComfyUI quantization contracts."
        )

    model_path = _get_models_base() / model_name
    if requested_download and not model_path.is_dir():
        logger.info("Downloading the user-selected official Fish Audio s2-pro checkpoint")
        _download_official_checkpoint(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Fish Audio s2-pro was not found at {model_path}. Select "
            f"'{HF_DEFAULT_MODEL_NAME}{AUTO_DOWNLOAD_SUFFIX}' to download it explicitly."
        )
    return model_path


def _find_decoder(model_dir: Path) -> Path:
    names = (
        "codec.pth",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        "decoder.pth",
        "vocoder.pth",
    )
    for directory in (model_dir, model_dir.parent):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Fish S2 codec checkpoint not found near {model_dir}")


@dataclass(frozen=True)
class FishS2Checkpoint:
    name: str
    path: Path
    dtype: torch.dtype


class FishS2CodecModel(nn.Module):
    """Give Comfy a settable top-level device owner around the third-party DAC."""

    def __init__(self, codec: nn.Module):
        super().__init__()
        self.codec = codec
        self.device = torch.device("cpu")

    @property
    def sample_rate(self):
        return self.codec.sample_rate

    @property
    def spec_transform(self):
        return self.codec.spec_transform

    def encode(self, *args, **kwargs):
        return self.codec.encode(*args, **kwargs)

    def from_indices(self, *args, **kwargs):
        return self.codec.from_indices(*args, **kwargs)


@dataclass
class FishS2Model:
    semantic: ModelPatcher
    codec: CoreModelPatcher
    tokenizer: object
    checkpoint: FishS2Checkpoint

    @property
    def device(self) -> torch.device:
        return self.semantic.load_device

    def semantic_memory_required(self, tokens: int) -> int:
        config = self.semantic.model.config
        slow = 2 * config.n_layer * config.n_local_heads * config.head_dim * tokens * 2
        fast = 2 * config.n_fast_layer * config.fast_n_local_heads * config.fast_head_dim * config.num_codebooks * 2
        return slow + fast + 256 * 1024 * 1024

    def load_semantic(self, tokens: int) -> None:
        memory_required = self.semantic_memory_required(tokens)
        model_management.load_models_gpu(
            [self.semantic],
            memory_required=memory_required,
            minimum_memory_required=memory_required,
            force_full_load=True,
        )

    def load_codec(self, audio_samples: int = 0) -> None:
        model_management.load_models_gpu(
            [self.codec], memory_required=audio_samples * 8 + 256 * 1024 * 1024
        )


def load_native_model(model_name: str) -> FishS2Model:
    load_device = model_management.get_torch_device()
    if not torch.cuda.is_available() or load_device.type != "cuda":
        raise RuntimeError("Native Fish Audio S2 currently requires CUDA with BF16 support.")

    model_path = resolve_model_path(model_name)
    from fish_speech.models.dac.inference import load_model as load_codec_model
    from fish_speech.models.text2semantic.llama import DualARTransformer

    semantic_model = DualARTransformer.from_pretrained(
        str(model_path),
        load_weights=True,
    )
    floating = (p for p in semantic_model.parameters() if p.is_floating_point())
    if any(parameter.dtype != torch.bfloat16 for parameter in floating):
        raise RuntimeError("The selected Fish Audio checkpoint is not the official BF16 s2-pro checkpoint.")

    codec_model = load_codec_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(_find_decoder(model_path)),
        device="cpu",
    )
    semantic_patcher = ModelPatcher(
        semantic_model,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
    )
    codec_patcher = CoreModelPatcher(
        FishS2CodecModel(codec_model),
        load_device=load_device,
        offload_device=model_management.vae_offload_device(),
    )
    return FishS2Model(
        semantic=semantic_patcher,
        codec=codec_patcher,
        tokenizer=semantic_model.tokenizer,
        checkpoint=FishS2Checkpoint(HF_DEFAULT_MODEL_NAME, model_path, torch.bfloat16),
    )
