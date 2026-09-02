"""Make the copied Stage-A project importable when tests run from workspace root."""

import sys
from pathlib import Path


STAGE_A_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_A_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_A_ROOT))

