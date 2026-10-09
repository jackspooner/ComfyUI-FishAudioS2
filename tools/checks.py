"""Run the unchanged native suite with real ComfyUI imports and CPU fixtures."""
import subprocess
import sys
from dev_environment import PYTHON, ROOT, test_environment

if len(sys.argv) > 1 and sys.argv[1] == "syntax":
    for folder in [ROOT, ROOT / "nodes", ROOT / "tests", ROOT / "tools"]:
        for source in folder.glob("*.py"):
            compile(source.read_bytes(), str(source), "exec")
else:
    result = subprocess.run([str(PYTHON), "-c", "from comfy.cli_args import args; args.cpu=True; import pytest; raise SystemExit(pytest.main(['-q']))"], cwd=ROOT, env=test_environment())
    raise SystemExit(result.returncode)
