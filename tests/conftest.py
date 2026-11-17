"""Make the spike importable and give it dummy config, so tests never need a key."""
import os
import sys
from pathlib import Path

# Must be set BEFORE `import agent`: the module reads them at import time.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENROUTER_MODEL", "test/model")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes" / "01-manual-loop"))
