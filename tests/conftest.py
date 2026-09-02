import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
COMFYUI = REPOSITORY.parents[1]
FISH_SOURCE = REPOSITORY / "fish_speech_src"

sys.path.insert(0, str(REPOSITORY))
sys.path.append(str(COMFYUI))
sys.path.append(str(FISH_SOURCE))
