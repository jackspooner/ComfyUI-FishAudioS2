"""Native modular Fish Audio S2-Pro ComfyUI nodes."""

from comfy_api.latest import io

from .loader import get_model_names, load_native_model
from .native_runtime import (
    decode_codes,
    encode_reference,
    generate_codes,
    make_conditioning,
    synthesize,
)

FishModelType = io.Custom("FISH_S2_MODEL")
FishConditioningType = io.Custom("FISH_S2_CONDITIONING")
FishReferenceType = io.Custom("FISH_S2_REFERENCE")
FishCodesType = io.Custom("FISH_S2_CODES")

LANGUAGES = [
    "auto", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de",
    "it", "tr", "nl", "sv", "no", "da", "fi", "pl", "hi", "vi", "th",
    "id", "ms", "uk", "bg", "hr", "cs", "sk", "sl", "ro", "hu", "et",
    "lv", "lt", "el", "he", "fa", "bn", "ta", "te", "kn", "ml", "si",
    "my", "km", "am", "ka", "az", "kk", "mn", "sw", "yo", "eu", "ca",
    "gl", "cy", "la", "sa", "ur", "ne", "tl", "jw",
]


def generation_inputs() -> list:
    return [
        io.Int.Input("seed", default=0, min=0, max=2**63 - 1),
        io.Int.Input("token_limit", default=0, min=0, max=32768, step=64),
        io.Int.Input("chunk_length", default=200, min=0, max=32768, step=10),
        io.Float.Input("temperature", default=0.8, min=0.01, max=1.99, step=0.05),
        io.Float.Input("top_p", default=0.8, min=0.01, max=1.0, step=0.05),
        io.Int.Input("top_k", default=30, min=0, max=4096),
        io.Float.Input("repetition_penalty", default=1.1, min=0.0, max=4.0, step=0.05),
    ]


class FishS2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2ModelLoader",
            display_name="Fish S2 Model Loader",
            category="audio/fish_s2",
            inputs=[io.Combo.Input("model", options=get_model_names())],
            outputs=[FishModelType.Output()],
        )

    @classmethod
    def execute(cls, model):
        return io.NodeOutput(load_native_model(model))


class FishS2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2Conditioning",
            display_name="Fish S2 Conditioning",
            category="audio/fish_s2",
            inputs=[
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", options=LANGUAGES, default="auto"),
            ],
            outputs=[FishConditioningType.Output()],
        )

    @classmethod
    def execute(cls, text, language):
        return io.NodeOutput(make_conditioning(text, language))


class FishS2EncodeReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2EncodeReference",
            display_name="Fish S2 Encode Reference",
            category="audio/fish_s2",
            inputs=[
                FishModelType.Input("model"),
                io.Audio.Input("audio"),
                io.String.Input("transcript", multiline=True),
                io.Int.Input("speaker", default=1, min=1, max=1024),
            ],
            outputs=[FishReferenceType.Output()],
        )

    @classmethod
    def execute(cls, model, audio, transcript, speaker):
        return io.NodeOutput(encode_reference(model, audio, transcript, speaker))


class FishS2GenerateCodes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        references = io.Autogrow.TemplatePrefix(
            input=FishReferenceType.Input("reference"), prefix="reference_", min=0
        )
        return io.Schema(
            node_id="FishS2GenerateCodes",
            display_name="Fish S2 Generate Codes",
            category="audio/fish_s2",
            inputs=[
                FishModelType.Input("model"),
                FishConditioningType.Input("conditioning"),
                io.Autogrow.Input("references", template=references, optional=True),
                *generation_inputs(),
            ],
            outputs=[FishCodesType.Output()],
        )

    @classmethod
    def execute(cls, model, conditioning, seed, token_limit, chunk_length, temperature, top_p, top_k, repetition_penalty, references=None):
        refs = [references[key] for key in sorted(references or {}) if references[key] is not None]
        return io.NodeOutput(generate_codes(model, conditioning, refs, seed, token_limit, chunk_length, temperature, top_p, top_k, repetition_penalty))


class FishS2DecodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2DecodeAudio",
            display_name="Fish S2 Decode Audio",
            category="audio/fish_s2",
            inputs=[
                FishModelType.Input("model"),
                FishCodesType.Input("codes"),
                io.Float.Input("inter_segment_pause", default=0.0, min=0.0, max=10.0, step=0.05),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, model, codes, inter_segment_pause):
        return io.NodeOutput(decode_codes(model, codes, inter_segment_pause))


class FishS2Synthesize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FishS2Synthesize",
            display_name="Fish S2 Synthesize",
            category="audio/fish_s2",
            inputs=[
                FishModelType.Input("model"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", options=LANGUAGES, default="auto"),
                io.Audio.Input("reference_audio", optional=True),
                io.String.Input("reference_transcript", multiline=True, optional=True, default=""),
                *generation_inputs(),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, model, text, language, seed, token_limit, chunk_length, temperature, top_p, top_k, repetition_penalty, reference_audio=None, reference_transcript=""):
        return io.NodeOutput(synthesize(model, text, language, seed, token_limit, chunk_length, temperature, top_p, top_k, repetition_penalty, reference_audio=reference_audio, reference_text=reference_transcript))
