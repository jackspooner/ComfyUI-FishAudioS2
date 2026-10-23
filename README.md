# ComfyUI Fish Audio S2

The isolated CPU development lock includes ComfyUI runtime packages (comfy-kitchen 0.2.39 and comfy-aimdo 0.5.6). Native readiness checks runtime requirements from the configured ComfyUI source; its frontend, workflow-template and embedded-document asset pins are excluded because model-free custom-node tests do not serve those assets. Their locked development copies remain unchanged. Setup does not alter the serving environment.

Native ComfyUI nodes for the official Fish Audio S2 Pro text-to-speech checkpoint, including voice references and multi-speaker dialogue.

## Upstream and this fork

This is a fork of [Saganaki22/ComfyUI-FishAudioS2](https://github.com/Saganaki22/ComfyUI-FishAudioS2); see its [upstream README](https://github.com/Saganaki22/ComfyUI-FishAudioS2/blob/master/README.md). The retained upstream v0.5.2 README describes Fish Audio S2 Pro TTS, reference voice cloning, emotion tags, multilingual synthesis, multi-speaker conversations and per-speaker tracks. Upstream also offers user-selected runtime/attention/precision policies and quantized variants. Those upstream capabilities are attributed to its authors; the fork's supported interface and limitations below differ. The upstream public URL returned 404 during this maintenance, so that summary uses the retained Git tag rather than claiming a fresh upstream release audit.

Our fork provides six composable native nodes, retains four legacy workflow adapters, delegates model lifetime to ComfyUI, and supports the official BF16 CUDA checkpoint. It also adds isolated CPU development preparation through native Python helpers; this development environment does not change the serving ComfyUI interpreter or certify full model inference. Fish Speech remains bundled. [LICENSE](LICENSE) and Fish Audio's [bundled source documentation](fish_speech_src/README.md) retain the upstream research-license terms and credit; per-speaker isolation credits the upstream project and the idea attributed there to [lazybuttalented](https://github.com/lazybuttalented).

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
git clone https://github.com/jackspooner/ComfyUI-FishAudioS2.git
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

For agent-managed model preparation, use Download Manager 2 with this explicit destination and verify the completed file set before loading. Follow its shared skill for authentication, persistent job IDs and recovery.

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

## Development preparation and checks

Run `python3 tools/setup_worktree.py` in a fresh Git CLI task worktree. No argument selects the full scope; `python3 tools/setup_worktree.py docs` selects documentation preparation while retaining the native suite during verification. The native command directly creates and checks the checkout-local CPU environment, restoring its committed lock when readiness fails. Git and a standalone Python 3.11+ interpreter are required; Linux x86_64/Python 3.13 is verified, and compatible version differences are reported before readiness checks. Other platforms are unverified.

Configure these nonsecret inputs outside Git:

```bash
export FISHS2_DEV_PYTHON=/absolute/standalone/python3.13
export COMFYUI_ROOT=/absolute/read-only/ComfyUI
python3 tools/setup_worktree.py
python3 tools/checks.py
```

`FISHS2_DEV_PYTHON` must be a standalone interpreter, not ComfyUI's serving virtualenv. Obtain it through the existing Python installation or the supported RuntimeHost/runtime installation workflow. `COMFYUI_ROOT` supplies native ComfyUI source and its requirements; setup never writes that checkout. The development environment is owned only by this worktree under ignored `.venv/dev-runtime/`. No dependency directory is copied from another checkout and no system-site packages are inherited.

Preparation checks tools, creates that local venv, restores `requirements-dev.lock.txt` with pip and CPU PyTorch wheels, and verifies real ComfyUI/Fish imports after restoration. Healthy repeats probe exact installed versions, declared requirements and required module files without repeatedly importing the heavy runtime. The lock includes this fork's declared runtime packages, native ComfyUI dependencies, pytest, Click for the bundled inference module, and the bundled Fish-compatible Transformers 4.57.3. Codec/audio-tool packages use the existing `--no-deps` policy. Installing this development lock never installs the bundled fish-speech package or alters the serving environment. Registry/PyTorch CPU-wheel access and disk space for an independent environment are required for cold setup; normal pip caching is supported. No model downloads, service starts or inference are part of setup.

Healthy repeats use native ownership, lock-version and required-module probes without reinstalling. If readiness fails, the native install action force-restores the committed lock; it does not advance versions. Fix the reported failure and rerun setup. To change dependencies intentionally, resolve them in an isolated development environment, validate the native suite, review and commit the regenerated lock together with this guidance.

Full validation is `python3 tools/checks.py` plus `python3 tools/checks.py syntax`. The first runs the unchanged `python -m pytest -q` suite through the local interpreter with real ComfyUI imports, offline Hugging Face/Transformers settings and `comfy.cli_args.args.cpu=True` before model-management import. Test assertions and collection are retained. The suite uses tiny synthetic configurations, checks node schemas, legacy translation, attention/masks, full Comfy model loading and codec → semantic → codec composition, and never loads the production checkpoint. Syntax checking compiles source in memory. These CPU checks do not establish CUDA/BF16 model readiness.

Repository-owned operating guidance is [fishaudio-comfyui-nodes](skills/fishaudio-comfyui-nodes/SKILL.md). Register only the stable canonical skills root after integration as documented in AGENTS.md.
