"""Makes the arbo_ocr package importable during test collection without
requiring the package to be pip-installed first. The project has no src/
layout and isn't installed in this environment, so pytest's own sys.path
insertion (which only adds the tests/ directory itself, since tests/ has no
__init__.py) isn't enough — the project root needs to be on sys.path too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
