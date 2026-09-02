# ComfyUI Fish Audio S2

Native ComfyUI nodes for the official Fish Audio S2 Pro text-to-speech checkpoint, including voice references and multi-speaker dialogue.

## Runtime model

The integration delegates runtime policy to ComfyUI:

- Fish's shipped attention, sampling, and cache behavior are preserved. ComfyUI patchers own model loading, eviction, unload, and reload instead of the old global engine cache and worker thread.
- The codec is a separate whole-model patcher.
- Codec encode, semantic generation, and codec decode are separate loading stages.
- Attention and sampling use the implementation shipped by the Fish custom-node package.
- ComfyUI interruption is checked in the token loop.
- Full-context causal-mask buffers are replaced by equivalent active-size masks, avoiding several gigabytes of fixed VRAM overhead.
- Unload Models and normal Comfy model eviction own model lifetime.

There are intentionally no node controls for device, precision, attention, compilation, CPU offload, or keeping models loaded.

This release supports the official `fishaudio/s2-pro` BF16 checkpoint on CUDA. FP8 and BitsAndBytes checkpoints fail with a message explaining that native quantized support is not available yet. CPU and MPS are not supported in this release.

## Installation

Install through ComfyUI Manager, or clone into `ComfyUI/custom_nodes` and install the declared dependencies in ComfyUI's Python environment:

```bash
git clone https://github.com/Saganaki22/ComfyUI-FishAudioS2.git
cd ComfyUI-FishAudioS2
python -m pip install -r requirements.txt
```

Fish Speech source is bundled. Do not install a separate `fish-speech` package over the ComfyUI environment.

## Model location

Place the official checkpoint in:

```text
ComfyUI/models/fishs2/s2-pro/
```

The folder must contain the semantic checkpoint, tokenizer/config files, and codec checkpoint. The loader also exposes `s2-pro (auto download)` when the local model is absent. Selecting that entry explicitly downloads `fishaudio/s2-pro`; loading a workflow never silently chooses another model.

Manual download example:

```bash
huggingface-cli download fishaudio/s2-pro --local-dir ComfyUI/models/fishs2/s2-pro
```

## Canonical modular graph

Basic synthesis:

```text
FishS2ModelLoader ───────────────┐
FishS2Conditioning ──────────────┼─> FishS2GenerateCodes ─> FishS2DecodeAudio
                                │
FishS2EncodeReference (optional)┘
```

The nodes are:

| Node ID | Output | Responsibility |
| --- | --- | --- |
| `FishS2ModelLoader` | `FISH_S2_MODEL` | Load the official local BF16 model as semantic and codec patchers. |
| `FishS2Conditioning` | `FISH_S2_CONDITIONING` | Prepare text/language and normalize speaker tags. |
| `FishS2EncodeReference` | `FISH_S2_REFERENCE` | Encode one Comfy `AUDIO` input to CPU VQ codes with transcript and speaker metadata. |
| `FishS2GenerateCodes` | `FISH_S2_CODES` | Generate CPU code segments from conditioning and zero or more autogrowing references. |
| `FishS2DecodeAudio` | `AUDIO` | Decode code segments and optionally insert silence between them. |
| `FishS2Synthesize` | `AUDIO` | Convenience composition for text and one optional reference. |

`FishS2GenerateCodes` owns sampling inputs: seed, token limit, chunk length, temperature, top-p, top-k, and repetition penalty.

### Voice cloning

Connect the same `FISH_S2_MODEL` to `FishS2EncodeReference` and `FishS2GenerateCodes`. Supply a transcript when possible. Reference speaker numbers are one-based in the UI.

### Multi-speaker text

Use one-based tags in authored text:

```text
[speaker_1]: Hello.
[speaker_2]: Welcome back.
```

`FishS2Conditioning` normalizes them to Fish's zero-based internal tags. Connect one encoded reference per speaker to the autogrowing reference input.

## Legacy workflow migration

These IDs remain registered for existing workflows:

- `FishS2TTS`
- `FishS2VoiceCloneTTS`
- `FishS2MultiSpeakerTTS`
- `FishS2MultiSpeakerSplitTTS`

They preserve their existing input schemas and outputs, but are deprecated adapters over the native runtime. Their old device, precision, attention, keep-loaded, offload, and compile inputs are ignored, with one warning per execution.

For new workflows, replace a legacy node with the modular graph above or `FishS2Synthesize`. The split adapter continues to provide a combined track plus ten per-speaker tracks.

## Development checks

The test suite uses tiny synthetic model configurations and never loads the production checkpoint:

```bash
python -m pytest -q
```

The tests cover node schemas, legacy registration/translation behavior, the shipped Fish attention policy, active-size masks, full Comfy model loading, and codec → semantic → codec composition.
