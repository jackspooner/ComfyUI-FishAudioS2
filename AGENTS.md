# FishAudio node development

Read README for upstream FishAudio attribution, bundled research licensing and supported node behavior.

Run `python3 tools/setup_worktree.py` (full/default or docs). Set `FISHS2_DEV_PYTHON` to an absolute standalone Python (3.13/Linux verified) and `COMFYUI_ROOT` to read-only native ComfyUI source. Setup owns only `.venv/dev-runtime` and restores the committed CPU lock when native readiness fails. It never inherits or modifies serving site-packages.

Run `python3 tools/checks.py` and `python3 tools/checks.py syntax`. Preserve the complete native pytest suite, real ComfyUI imports and CPU fixtures. CPU tests do not certify CUDA inference.

Readiness checks runtime requirements, excluding ComfyUI frontend/template/docs assets unused by CPU-node tests. The development lock pins comfy-kitchen 0.2.39 and comfy-aimdo 0.5.6. ComfyUI owns models, queue, interruption and GPU lifetime. Setup starts no services/models; avoid the plugin root auto-installer for development setup. Authoritative node guidance lives in `skills/fishaudio-comfyui-nodes/`.
