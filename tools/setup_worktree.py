"""Prepare this checkout through the installed DevTasks owner."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
try:
    result = subprocess.run(["devtask", "setup", "--repo", str(root), "--scope", sys.argv[1] if len(sys.argv) > 1 else "full", "--json"])
except FileNotFoundError:
    raise SystemExit("Install Workspace2 DevTasks and configure its owners; see README.md.")
raise SystemExit(result.returncode)
