"""Locations for the V1.2 fitting and timing drivers."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
CODE_DIR = REPOSITORY / "exp" / "robin"
RAW_DATA = Path(os.environ.get("TURNPIKE_RAW_DATA", REPOSITORY / "Raw Data")).expanduser().resolve()
ARTIFACTS = Path(os.environ.get("TURNPIKE_OUTPUT", REPOSITORY / "artifacts")).expanduser().resolve()

def output_directory(group: str) -> Path:
    result = ARTIFACTS / group
    result.mkdir(parents=True, exist_ok=True)
    return result
