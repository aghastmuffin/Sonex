"""Double-click launcher for Sonex Setup (no console window on Windows)."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
runpy.run_path(str(Path(__file__).resolve().parent / "setup_gui.py"), run_name="__main__")
