# conftest.py

import sys
import os

# Compute absolute path to the project root (this file’s directory)
ROOT = os.path.dirname(__file__)
# Path to the `src/` folder
SRC  = os.path.join(ROOT, "src")

# Prepend `src/` to sys.path so that
# `import src.backend.repositories...` works
if SRC not in sys.path:
    sys.path.insert(0, SRC)
