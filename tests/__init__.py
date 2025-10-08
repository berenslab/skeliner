from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Ensure Matplotlib (pulled indirectly by trimesh during tests) has a writable
# cache directory even inside read-only home environments.
_mpl_cache = Path(tempfile.gettempdir()) / "skeliner-mpl-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
