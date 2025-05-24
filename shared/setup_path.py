# shared/setup_path.py
import sys
from pathlib import Path

def extend_path():
    shared_path = Path(__file__).resolve().parents[1]
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
