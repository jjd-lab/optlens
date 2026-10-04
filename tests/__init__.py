import sys
from pathlib import Path

# the editable install's .pth file is skipped by Python 3.13 when macOS flags it hidden (as sandboxed installs leave it)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
