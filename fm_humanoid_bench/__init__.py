"""Foundation-model humanoid Brain × Body benchmark."""
from pathlib import Path
import sys

# The simulator is a backend dependency, kept outside the benchmark package.
_SIMULATOR = Path(__file__).resolve().parents[1] / "simulator"
if str(_SIMULATOR) not in sys.path:
    sys.path.insert(0, str(_SIMULATOR))
