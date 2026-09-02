"""Shared synchronous Fish S2 execution used by native and legacy nodes."""

import logging
import re
import time
from dataclasses import dataclass

import torch
import torchaudio

import comfy.model_management as model_management

from .loader import FishS2Model

logger = logging.getLogger("FishAudioS2")


@dataclass(frozen=True)
class FishS2ConditioningData:
    text: str
    language: str


@dataclass(frozen=True)
class FishS2ReferenceData:
    codes: torch.Tensor
    transcript: str
    speaker: int


@dataclass(frozen=True)
class FishS2CodesData:
    segments: tuple[torch.Tensor, ...]
    texts: tuple[str, ...]
    speakers: tuple[int | None, ...]


def _codec_sample_rate(codec) -> int:
    if hasattr(codec, "spec_transform"):
        return int(codec.spec_transform.sample_rate)
    return int(codec.sample_rate)


def normalize_speaker_tags(text: str) -> str:
    def replace(match: re.Match) -> str:
        speaker = int(match.group(1))
        if speaker < 1:
            raise ValueError("Fish speaker numbers are one-based and must be at least 1.")
        return f"<|speaker:{speaker - 1}|>"

    return re.sub(r"\[speaker_(\d+)\]\s*:\s*", replace, text, flags=re.IGNORECASE)


def make_conditioning(text: str, language: str) -> FishS2ConditioningData:
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    normalized = normalize_speaker_tags(text.strip())
    if language != "auto":
        normalized = f"[{language}] {normalized}"
    return FishS2ConditioningData(normalized, language)


def encode_reference(
    model: FishS2Model,
    audio: dict,
    transcript: str,
    speaker: int,
) -> FishS2ReferenceData:
    model_management.throw_exception_if_processing_interrupted()
    waveform = audio["waveform"][:1].mean(dim=1, keepdim=True)
    sample_rate = int(audio["sample_rate"])
    codec = model.codec.model
    target_rate = _codec_sample_rate(codec)
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)

    model.load_codec(waveform.shape[-1])
    dtype = next(codec.parameters()).dtype
    waveform = waveform.to(device=model.codec.load_device, dtype=dtype)
    lengths = torch.tensor([waveform.shape[-1]], device=waveform.device, dtype=torch.long)
    indices, feature_lengths = codec.encode(waveform, lengths)
    codes = indices[0, :, : int(feature_lengths[0])].to(device="cpu", dtype=torch.long).contiguous()
    return FishS2ReferenceData(codes, transcript.strip(), speaker)


def generate_codes(
    model: FishS2Model,
    conditioning: FishS2ConditioningData,
    references: list[FishS2ReferenceData],
    seed: int,
    token_limit: int,
    chunk_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> FishS2CodesData:
    from fish_speech.models.text2semantic.inference import decode_one_token_ar, generate_long
    from fish_speech.utils import set_seed

    set_seed(seed)
    semantic = model.semantic.model
    prompt_tokens = [reference.codes for reference in references]
    prompt_text = [f"<|speaker:{reference.speaker - 1}|>{reference.transcript}" for reference in references]
    prompt_size = sum(codes.shape[-1] for codes in prompt_tokens)
    requested_tokens = token_limit if token_limit > 0 else semantic.config.max_seq_len
    model.load_semantic(min(semantic.config.max_seq_len, prompt_size + requested_tokens + 256))
    semantic._codebook_pos = [
        torch.tensor([index], device=model.device, dtype=torch.long)
        for index in range(semantic.config.num_codebooks)
    ]

    segments = []
    texts = []
    speakers = []
    started = time.perf_counter()
    try:
        for response in generate_long(
            model=semantic,
            device=model.device,
            decode_one_token=decode_one_token_ar,
            text=conditioning.text,
            max_new_tokens=token_limit,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            interruption_check=model_management.throw_exception_if_processing_interrupted,
        ):
            model_management.throw_exception_if_processing_interrupted()
            if response.action != "sample":
                continue
            codes = response.codes.to(device="cpu", dtype=torch.long).contiguous()
            segments.append(codes)
            texts.append(response.text or "")
            match = re.search(r"<\|speaker:(\d+)\|>", response.text or "")
            speakers.append(int(match.group(1)) + 1 if match else None)
    finally:
        prompt_tokens.clear()
        del semantic._codebook_pos

    if not segments:
        raise RuntimeError("Fish S2 did not generate any audio codes.")
    frames = sum(segment.shape[-1] for segment in segments)
    elapsed = time.perf_counter() - started
    logger.info("Generated %d Fish S2 code frames in %.1fs (%.1f frames/s)", frames, elapsed, frames / elapsed)
    return FishS2CodesData(tuple(segments), tuple(texts), tuple(speakers))


def decode_codes(model: FishS2Model, codes: FishS2CodesData, pause: float) -> dict:
    codec = model.codec.model
    estimated_samples = sum(segment.shape[-1] for segment in codes.segments) * 2048
    model.load_codec(estimated_samples)
    sample_rate = _codec_sample_rate(codec)
    parts = []
    for index, segment in enumerate(codes.segments):
        model_management.throw_exception_if_processing_interrupted()
        audio = codec.from_indices(segment.unsqueeze(0).to(model.codec.load_device))[0, 0]
        parts.append(audio.float().cpu())
        if pause > 0 and index + 1 < len(codes.segments):
            parts.append(torch.zeros(round(pause * sample_rate), dtype=torch.float32))
    waveform = torch.cat(parts).view(1, 1, -1)
    return {"waveform": waveform, "sample_rate": sample_rate}


def synthesize(
    model: FishS2Model,
    text: str,
    language: str,
    seed: int,
    token_limit: int,
    chunk_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    pause: float = 0.0,
    reference_audio: dict | None = None,
    reference_text: str = "",
    speaker: int = 1,
) -> dict:
    references = []
    if reference_audio is not None:
        references.append(encode_reference(model, reference_audio, reference_text, speaker))
    codes = generate_codes(
        model,
        make_conditioning(text, language),
        references,
        seed,
        token_limit,
        chunk_length,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )
    return decode_codes(model, codes, pause)
