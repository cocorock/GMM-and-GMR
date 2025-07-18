# test_environment.py
import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print(f"\nPYTHONPATH environment variable: {os.environ.get('PYTHONPATH', 'Not set')}")

# Test imports
try:
    import numpy
    print("✓ numpy works")
except ImportError:
    print("❌ numpy not found")

try:

    from tpgmm.utils.file_system import load_txt
    from tpgmm.utils.casting import ssv_to_ndarray
    from tpgmm.utils.plot.plot import plot_trajectories
    import numpy as np
    from glob import glob
    from numpy import ndarray

    print("✓ TPGMM import works")
except ImportError as e:
    print(f"❌ TPGMM import failed: {e}")