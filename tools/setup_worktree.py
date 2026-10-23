"""Restore only this checkout's native development environment."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
scope = sys.argv[1] if len(sys.argv) == 2 else "full"
if len(sys.argv) > 2 or scope not in ("full", "docs"):
    raise SystemExit("Usage: python3 tools/setup_worktree.py [full|docs]")
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required.")
subprocess.run(["git", "--version"], check=True)

helper = root / "tools/dev_environment.py"
def run(action, check=True):
    return subprocess.run([sys.executable, str(helper), action], cwd=root, check=check)

run("prerequisites")
run("create")
if run("installed", check=False).returncode:
    run("install")
