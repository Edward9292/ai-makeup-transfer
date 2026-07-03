"""Pytest path setup: the project uses a plain src/ layout without an installed package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
