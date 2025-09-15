
# Add custom package directory to PYTHONPATH
import sys
import os
custom_path = "/storage/scratch1/6/cphan36/python_packages/lib/python3.10/site-packages"
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)
    print(f"Added {custom_path} to Python path")
