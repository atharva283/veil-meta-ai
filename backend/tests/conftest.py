import os
import sys
import tempfile
from pathlib import Path

# Tests always run offline against the mocks, in a throwaway data dir.
os.environ["VEIL_FORCE_MOCK"] = "planner,segmenter,auditor"
os.environ["VEIL_DATA_DIR"] = tempfile.mkdtemp(prefix="veil-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
