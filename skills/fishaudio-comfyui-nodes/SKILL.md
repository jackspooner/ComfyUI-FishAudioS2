---
name: fishaudio-comfyui-nodes
description: Prepare and verify this FishAudio ComfyUI fork, discover its native synthesis nodes, and preserve ComfyUI-owned model loading, references and legacy workflow compatibility.
---

# FishAudio ComfyUI nodes

This fork implements six modular nodes and four legacy adapters; inspect current schemas through the selected ComfyUI instance's `/object_info` before authoring a workflow. The upstream README and this fork's supported behavior differ; [README](../../README.md) documents attribution and current limitations. Loading a registered node does not establish model readiness.

## Model operation

Use the official `fishaudio/s2-pro` BF16 checkpoint on CUDA. This fork rejects FP8/BitsAndBytes models, CPU and MPS for production inference. Discover the selected instance's configured model roots and loader choices; the default category is `fishs2/s2-pro`. Prepare all semantic/tokenizer/config and codec files through Download Manager 2 with explicit destinations. The loader has an explicit auto-download choice, but agent model downloads follow the shared download owner. Never substitute a quantized or incomplete checkpoint to make a graph run.

`FishS2ModelLoader` returns `FISH_S2_MODEL`; share it with `FishS2EncodeReference`, `FishS2GenerateCodes` and decode. `FishS2Conditioning` prepares text/language and normalizes authored one-based speaker tags. An optional Comfy `AUDIO` reference with transcript and one-based speaker number becomes CPU VQ reference codes; connect one reference per speaker to the generator's autogrowing inputs. The generator owns seed/token limit/chunk length/temperature/top-p/top-k/repetition penalty and returns CPU code segments. `FishS2DecodeAudio` produces Comfy `AUDIO`; `FishS2Synthesize` composes the same stages for one optional reference. Concrete use cases are text-only speech, reference voice cloning and multi-speaker dialogue.

ComfyUI owns patcher loading/eviction/unload/reload and interruption. Codec encode, semantic generation and codec decode are separate stages. Do not add device/precision/attention/keep-loaded controls to the modular interface. Legacy `FishS2TTS`, `FishS2VoiceCloneTTS`, `FishS2MultiSpeakerTTS` and `FishS2MultiSpeakerSplitTTS` retain schemas/outputs; old runtime-policy inputs are ignored with a deprecation warning. Split output retains a combined track and ten speaker tracks. Confirm errors and returned execution IDs before retrying an uncertain submission; never replay work solely because a client disconnected. CUDA memory/model completeness remain real prerequisites, and CPU tests do not verify full inference or quantify GPU cost.

## Isolated development

Configure standalone `FISHS2_DEV_PYTHON` and read-only native `COMFYUI_ROOT`, then run `python3 tools/setup_worktree.py` (full by default) or append `docs`. Installed DevTasks and private host owner configuration are prerequisites. The adapter creates only checkout-local `.venv/dev-runtime`, restores the committed CPU development lock and records ignored `.cutlery/devtask/` evidence. It never inherits/changes serving site-packages, starts services or downloads models. Linux x86_64/Python 3.13 is verified; compatible differences are checked and other platforms remain unverified.

Healthy repeats perform zero installs/builds. Missing outputs or changed lock/setup inputs invalidate the affected preparation; node source edits preserve healthy installation. Inspect the named action log, repair the prerequisite, then rerun; do not copy another venv or reset shared state. Full verification always runs `python3 tools/checks.py` and `python3 tools/checks.py syntax`, preserving native pytest assertions and real ComfyUI imports with CPU fixture initialization/offline model settings. Register this skill only from canonical main after integration through SharedAgentSkills, preserving existing exclusions. Follow README for dependency-lock updates, licenses and bootstrap requirements.
