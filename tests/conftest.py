"""pytest configuration: add the package root to sys.path."""

import sys
from pathlib import Path

# Make the specificity_index package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
