"""Practice 7: reproducible recommendation experiment."""
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / '.cache'
os.environ.setdefault('SURPRISE_DATA_FOLDER', str(_cache / 'surprise'))
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
