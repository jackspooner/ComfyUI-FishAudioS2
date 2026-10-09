"""Own only the checkout-local CPU development environment; never the serving venv."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv/dev-runtime"
PYTHON = ENVIRONMENT / "bin/python"


def run(command, capture=False, **kwargs):
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture, **kwargs)


def configuration():
    base = os.environ.get("FISHS2_DEV_PYTHON", "")
    comfy = os.environ.get("COMFYUI_ROOT", "")
    if not base or not Path(base).is_absolute() or not Path(base).is_file():
        raise SystemExit("Set FISHS2_DEV_PYTHON to an absolute standalone Python 3.13 executable; never the serving venv.")
    if not comfy or not Path(comfy).is_absolute() or not (Path(comfy) / "comfy/model_management.py").is_file():
        raise SystemExit("Set COMFYUI_ROOT to the read-only native ComfyUI source checkout.")
    evidence = json.loads(run([base, "-c", "import json,sys; print(json.dumps({'version':list(sys.version_info[:3]),'prefix':sys.prefix,'basePrefix':sys.base_prefix}))"], True).stdout)
    if evidence["prefix"] != evidence["basePrefix"]:
        raise SystemExit("FISHS2_DEV_PYTHON must be a standalone interpreter; a serving/shared virtualenv is refused.")
    if tuple(evidence["version"][:2]) < (3, 11):
        raise SystemExit("Python 3.11+ required; Python 3.13 is verified.")
    return base, Path(comfy), evidence


def test_environment():
    _, comfy, _ = configuration()
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [ROOT, comfy, ROOT / "fish_speech_src"])),
                PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def dependency_probe():
    # The bundled project's training dependencies are not the node runtime.
    return """
import importlib.metadata as metadata
from pathlib import Path
import tomllib
from packaging.requirements import Requirement

def verify(lines):
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            version = metadata.version(requirement.name)
            assert version in requirement.specifier, f'{requirement}: installed {version}; refresh the development lock deliberately'

verify(Path('requirements-dev.lock.txt').read_text().splitlines())
verify(Path('requirements.txt').read_text().splitlines())
verify(tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies'])
verify(Path(__import__('os').environ['COMFYUI_ROOT'], 'requirements.txt').read_text().splitlines())
import torch, torchaudio, torchvision, pytest, transformers, click, dac, audiotools
assert torch.version.cuda is None, 'CPU development wheels required'
from comfy.cli_args import args
args.cpu = True
import comfy.model_management
from fish_speech.models.text2semantic import llama, inference
import nodes.native_runtime
"""


def main(action):
    base, comfy, evidence = configuration()
    if action in {"versions", "test-versions"}:
        versions = {"base":base,"python":evidence,"comfyHead":run(["git","-C",str(comfy),"rev-parse","HEAD"],True).stdout.strip(),"comfyRequirements":hashlib.sha256((comfy/"requirements.txt").read_bytes()).hexdigest()}
        if action == "test-versions":
            sources = sorted((comfy / "comfy").rglob("*.py")) + [comfy / "folder_paths.py"]
            versions["comfySources"] = hashlib.sha256(b"".join(str(path.relative_to(comfy)).encode() + b"\0" + path.read_bytes() for path in sources)).hexdigest()
        print(json.dumps(versions,sort_keys=True))
    elif action == "prerequisites":
        run(["git", "--version"])
        if evidence["version"][:2] != [3, 13]:
            print(f"Expected Python 3.13; found {evidence['version']}. Readiness and native tests determine compatibility.")
        if not (comfy / "requirements.txt").is_file():
            raise SystemExit("ComfyUI requirements.txt is missing; restore the configured source checkout.")
    elif action == "create":
        if ENVIRONMENT.is_symlink():
            raise SystemExit("Development environment must not be a symlink or shared writable environment.")
        if not PYTHON.is_file() or not (ENVIRONMENT / "pyvenv.cfg").is_file():
            run([base, "-m", "venv", str(ENVIRONMENT)])
        main("environment-ready")
    elif action == "environment-ready":
        run([str(PYTHON), "-c", "import pathlib,sys; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve(), 'Development interpreter must own this checkout-local venv'; assert pathlib.Path(sys.base_prefix).resolve()==pathlib.Path(sys.argv[2]).resolve(), 'Development venv uses a different base; recreate only this checkout-local environment'", str(ENVIRONMENT), evidence["basePrefix"]])
    elif action == "install":
        main("environment-ready")
        run([str(PYTHON), "-m", "pip", "install", "--force-reinstall", "--no-deps", "--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", "requirements-dev.lock.txt"])
        main("ready")
    elif action == "ready":
        main("environment-ready")
        run([str(PYTHON), "-c", dependency_probe()], env=test_environment())
    else:
        raise SystemExit(f"Unknown development preparation action: {action}")


if __name__ == "__main__":
    main(sys.argv[1])
