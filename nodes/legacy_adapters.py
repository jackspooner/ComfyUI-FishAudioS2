"""Deprecated node IDs backed by the native Fish S2 runtime."""

import logging
import re

import torch
from comfy_api.latest import io

from .loader import get_model_names, load_native_model
from .native_nodes import LANGUAGES
from .native_runtime import synthesize

logger = logging.getLogger("FishAudioS2")
MAX_SPEAKERS = 10

COMMON_GENERATION_INPUTS = {
    "max_new_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
    "chunk_length": ("INT", {"default": 200, "min": 100, "max": 400, "step": 10}),
    "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
    "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
    "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.9, "max": 2.0, "step": 0.05}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
}


def _legacy_runtime_inputs() -> dict:
    return {
        "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
        "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
        "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {"default": "auto"}),
        "keep_model_loaded": ("BOOLEAN", {"default": True}),
        "offload_to_cpu": ("BOOLEAN", {"default": False}),
        "compile_model": ("BOOLEAN", {"default": False}),
    }


def _warn(node_id: str) -> None:
    logger.warning(
        "%s is deprecated; use the modular Fish S2 nodes. Legacy device, precision, "
        "attention, loading, offload, and compile controls are ignored.",
        node_id,
    )


def _synthesis_kwargs(values: dict) -> dict:
    return {
        "seed": values["seed"],
        "token_limit": values["max_new_tokens"],
        "chunk_length": values["chunk_length"],
        "temperature": values["temperature"],
        "top_p": values["top_p"],
        "top_k": 30,
        "repetition_penalty": values["repetition_penalty"],
    }


class FishS2TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": (get_model_names(), {}),
            "text": ("STRING", {"multiline": True, "default": "Hello! [excited] This is Fish Audio S2."}),
            "language": (LANGUAGES, {"default": "auto"}),
            "device": _legacy_runtime_inputs()["device"],
            "precision": _legacy_runtime_inputs()["precision"],
            "attention": _legacy_runtime_inputs()["attention"],
            **COMMON_GENERATION_INPUTS,
            "keep_model_loaded": _legacy_runtime_inputs()["keep_model_loaded"],
            "offload_to_cpu": _legacy_runtime_inputs()["offload_to_cpu"],
            "compile_model": _legacy_runtime_inputs()["compile_model"],
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FishAudioS2/deprecated"

    def generate(self, model_path, text, language, device, precision, attention, max_new_tokens, chunk_length, temperature, top_p, repetition_penalty, seed, keep_model_loaded, offload_to_cpu, compile_model):
        _warn("FishS2TTS")
        values = locals()
        model = load_native_model(model_path)
        return (synthesize(model, text, language, **_synthesis_kwargs(values)),)


class FishS2VoiceCloneTTS(FishS2TTS):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()["required"]
        required = {
            "model_path": base.pop("model_path"),
            "text": base.pop("text"),
            "reference_audio": ("AUDIO", {}),
            **base,
        }
        return {"required": required, "optional": {"reference_text": ("STRING", {"multiline": True, "default": ""})}}

    def generate(self, model_path, text, reference_audio, language, device, precision, attention, max_new_tokens, chunk_length, temperature, top_p, repetition_penalty, seed, keep_model_loaded, offload_to_cpu, compile_model, reference_text=""):
        _warn("FishS2VoiceCloneTTS")
        values = locals()
        model = load_native_model(model_path)
        return (synthesize(model, text, language, reference_audio=reference_audio, reference_text=reference_text, **_synthesis_kwargs(values)),)


def _dialogue(text: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r"\[speaker_(\d+)\]\s*:\s*", text, re.IGNORECASE))
    turns = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turns.append((int(match.group(1)), text[match.end():end].strip()))
    return [(speaker, line) for speaker, line in turns if line]


def _run_dialogue(node_id: str, split: bool, **values):
    _warn(node_id)
    model = load_native_model(values["model_path"])
    turns = _dialogue(values["text"])
    if not turns:
        raise ValueError("No speaker lines found. Use [speaker_1]: text format.")

    rendered = []
    speakers = []
    sample_rate = None
    for line_index, (speaker, line) in enumerate(turns):
        if speaker > values["num_speakers"]:
            raise ValueError(f"Speaker {speaker} exceeds num_speakers={values['num_speakers']}.")
        reference = values.get(f"speaker_{speaker}_audio")
        if reference is None:
            raise ValueError(f"Missing reference audio: speaker_{speaker}_audio")
        audio = synthesize(
            model,
            line,
            values["language"],
            reference_audio=reference,
            reference_text=values.get(f"speaker_{speaker}_ref_text", ""),
            speaker=speaker,
            **{**_synthesis_kwargs(values), "seed": values["seed"] + line_index},
        )
        rendered.append(audio["waveform"][0, 0])
        speakers.append(speaker)
        sample_rate = audio["sample_rate"]

    pause = torch.zeros(round(values["pause_after_speaker"] * sample_rate))
    combined_parts = []
    for index, waveform in enumerate(rendered):
        combined_parts.append(waveform)
        if index + 1 < len(rendered) and pause.numel():
            combined_parts.append(pause)
    combined = torch.cat(combined_parts)
    combined_audio = {"waveform": combined.view(1, 1, -1), "sample_rate": sample_rate}
    if not split:
        return (combined_audio,)

    tracks = []
    for selected in range(1, MAX_SPEAKERS + 1):
        parts = []
        for index, waveform in enumerate(rendered):
            parts.append(waveform if speakers[index] == selected else torch.zeros_like(waveform))
            if index + 1 < len(rendered) and pause.numel():
                parts.append(torch.zeros_like(pause))
        tracks.append({"waveform": torch.cat(parts).view(1, 1, -1), "sample_rate": sample_rate})
    return (combined_audio, *tracks)


def _speaker_inputs(count: int) -> list:
    inputs = []
    for speaker in range(1, count + 1):
        inputs.append(io.Audio.Input(f"speaker_{speaker}_audio", optional=True))
        inputs.append(io.String.Input(f"speaker_{speaker}_ref_text", default="", optional=True))
    return inputs


def _legacy_multi_inputs() -> list:
    return [
        io.Combo.Input("model_path", options=get_model_names()),
        io.String.Input("text", multiline=True, default="[speaker_1]: Hello.\n[speaker_2]: Hi!"),
        io.Combo.Input("language", options=LANGUAGES, default="auto"),
        io.Combo.Input("device", options=["auto", "cuda", "cpu", "mps"], default="auto"),
        io.Combo.Input("precision", options=["auto", "bfloat16", "float16", "float32"], default="auto"),
        io.Combo.Input("attention", options=["auto", "sdpa", "sage_attention", "flash_attention"], default="auto"),
        io.Int.Input("max_new_tokens", default=0, min=0, max=4096, step=64),
        io.Int.Input("chunk_length", default=200, min=100, max=400, step=10),
        io.Float.Input("temperature", default=0.8, min=0.1, max=1.0, step=0.05),
        io.Float.Input("top_p", default=0.8, min=0.1, max=1.0, step=0.05),
        io.Float.Input("repetition_penalty", default=1.1, min=0.9, max=2.0, step=0.05),
        io.Int.Input("seed", default=0, min=0, max=2**31 - 1),
        io.Boolean.Input("keep_model_loaded", default=True),
        io.Boolean.Input("offload_to_cpu", default=False),
        io.Boolean.Input("compile_model", default=False),
        io.Float.Input("pause_after_speaker", default=0.4, min=0.0, max=5.0, step=0.1),
        io.DynamicCombo.Input(
            "num_speakers",
            display_name="Number of Speakers",
            options=[
                io.DynamicCombo.Option(key=str(count), inputs=_speaker_inputs(count))
                for count in range(2, MAX_SPEAKERS + 1)
            ],
        ),
    ]


class FishS2MultiSpeakerTTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2MultiSpeakerTTS",
            display_name="Fish S2 Multi-Speaker TTS (Deprecated)",
            category="FishAudioS2/deprecated",
            inputs=_legacy_multi_inputs(),
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, num_speakers, **kwargs):
        return io.NodeOutput(*_run_dialogue("FishS2MultiSpeakerTTS", False, num_speakers=int(num_speakers), **kwargs))


class FishS2MultiSpeakerSplitTTS(FishS2MultiSpeakerTTS):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2MultiSpeakerSplitTTS",
            display_name="Fish S2 Multi-Speaker Split TTS (Deprecated)",
            category="FishAudioS2/deprecated",
            inputs=_legacy_multi_inputs(),
            outputs=[io.Audio.Output(display_name="audio")] + [
                io.Audio.Output(display_name=f"speaker_{speaker}_audio")
                for speaker in range(1, MAX_SPEAKERS + 1)
            ],
        )

    @classmethod
    def execute(cls, num_speakers, **kwargs):
        return io.NodeOutput(*_run_dialogue("FishS2MultiSpeakerSplitTTS", True, num_speakers=int(num_speakers), **kwargs))
