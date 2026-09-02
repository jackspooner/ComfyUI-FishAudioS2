import logging
from types import SimpleNamespace

import torch

from nodes import legacy_adapters, loader, native_nodes, native_runtime


INFRASTRUCTURE_INPUTS = {
    "device", "precision", "attention", "keep_model_loaded", "offload_to_cpu", "compile_model"
}


def test_native_schemas_have_no_runtime_policy_controls():
    classes = (
        native_nodes.FishS2ModelLoader,
        native_nodes.FishS2Conditioning,
        native_nodes.FishS2EncodeReference,
        native_nodes.FishS2GenerateCodes,
        native_nodes.FishS2DecodeAudio,
        native_nodes.FishS2Synthesize,
    )
    for node in classes:
        input_ids = {item.id for item in node.define_schema().inputs}
        assert not input_ids & INFRASTRUCTURE_INPUTS


def test_legacy_ids_preserve_runtime_inputs_and_warn_once(monkeypatch, caplog):
    schema = legacy_adapters.FishS2TTS.INPUT_TYPES()["required"]
    assert INFRASTRUCTURE_INPUTS <= schema.keys()
    monkeypatch.setattr(legacy_adapters, "load_native_model", lambda _: object())
    monkeypatch.setattr(legacy_adapters, "synthesize", lambda *args, **kwargs: {"waveform": None, "sample_rate": 1})
    with caplog.at_level(logging.WARNING, logger="FishAudioS2"):
        legacy_adapters.FishS2TTS().generate(
            "s2-pro", "hello", "auto", "cpu", "float32", "sdpa",
            16, 200, 0.8, 0.8, 1.1, 1, False, True, True,
        )
    messages = [record.message for record in caplog.records if "deprecated" in record.message]
    assert len(messages) == 1


def test_synthesize_composes_codec_semantic_codec(monkeypatch):
    order = []
    reference = object()
    codes = object()
    monkeypatch.setattr(native_runtime, "encode_reference", lambda *args: order.append("codec") or reference)
    monkeypatch.setattr(native_runtime, "generate_codes", lambda *args: order.append("semantic") or codes)
    monkeypatch.setattr(native_runtime, "decode_codes", lambda *args: order.append("codec") or {"audio": True})
    result = native_runtime.synthesize(
        object(), "hello", "auto", 1, 8, 200, 0.8, 0.8, 30, 1.1,
        reference_audio={"waveform": object()},
    )
    assert result == {"audio": True}
    assert order == ["codec", "semantic", "codec"]


def test_speaker_tags_are_normalized_to_zero_based_internal_tags():
    conditioning = native_runtime.make_conditioning(
        "[speaker_1]: Hello\n[speaker_3]: Welcome", "auto"
    )
    assert conditioning.text == "<|speaker:0|>Hello\n<|speaker:2|>Welcome"


def test_semantic_model_requests_full_comfy_managed_loading(monkeypatch):
    calls = []
    semantic = SimpleNamespace(
        load_device=torch.device("cuda"),
        model=SimpleNamespace(
            config=SimpleNamespace(
                n_layer=2,
                n_local_heads=2,
                head_dim=4,
                n_fast_layer=1,
                fast_n_local_heads=2,
                fast_head_dim=4,
                num_codebooks=2,
            )
        ),
        model_size=lambda: 8 * 1024**3,
    )
    model = loader.FishS2Model(semantic, object(), object(), object())
    monkeypatch.setattr(loader.model_management, "load_models_gpu", lambda *args, **kwargs: calls.append(kwargs))
    model.load_semantic(256)
    assert calls[0]["force_full_load"] is True
