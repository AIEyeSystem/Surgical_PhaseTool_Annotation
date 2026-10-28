import os
from enum import Enum
from pathlib import Path
from IVAS.utils.persistence import atomic_write_bytes

import yaml

from IVAS.branding import CATEGORY_CONFIG_NAME, LEGACY_CATEGORY_CONFIG_NAMES

__all__ = [
    "load_config",
    "save_config",
    "STATUSMode",
    "DRAWMode",
    "MAPMode",
    "CONTOURMode",
]

IVAS_ROOT = os.path.split(os.path.abspath(__file__))[0]
"""Project root - IVAS/"""
# Source checkouts retain portable paths; installed/read-only packages use user data.
# Importing this module never creates directories or modifies package resources.
_package_root = Path(IVAS_ROOT)
_installed = any(part in {"site-packages", "dist-packages"} for part in _package_root.parts)
_user_root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "IVAS"
DATA_ROOT = Path(os.environ.get("IVAS_DATA_DIR") or (
    _user_root if _installed or not os.access(IVAS_ROOT, os.W_OK) else _package_root))
SOFTWARE_CONFIG_FILE = str(DATA_ROOT / "software.yaml")
CONFIG_FILE = str(DATA_ROOT / CATEGORY_CONFIG_NAME)
CHECKPOINT_PATH = str(DATA_ROOT / "checkpoints")


def load_config(file: str) -> dict:
    r"""
    Load config file

    Arguments:
        file (str): config file path

    Returns:
        dict: config dict
    """
    path = Path(file)
    if not path.is_file():
        candidates = []
        if path.name == CATEGORY_CONFIG_NAME:
            candidates.extend(path.with_name(name) for name in LEGACY_CATEGORY_CONFIG_NAMES)
        if path.parent == DATA_ROOT:
            candidates.append(_package_root / path.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if not path.is_file():
            return {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ValueError("Invalid YAML in {}: {}".format(path, error)) from error
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping: {}".format(path))
    return cfg


def save_config(cfg: dict, file: str) -> None:
    """
    Save config file

    Arguments:
        cfg (dict): config dict
        file (str): config file path
    """
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    s = yaml.safe_dump(cfg, allow_unicode=True)
    atomic_write_bytes(file, s.encode("utf-8"))


class STATUSMode(Enum):
    """The status mode"""

    VIEW = 0
    CREATE = 1
    EDIT = 2
    REPAINT = 3


class DRAWMode(Enum):
    """The draw mode."""

    POLYGON = 0
    """Manually draw polygon"""
    SEGMENTANYTHING_POINT = 1
    """Segment anything with point prompt."""
    SEGMENTANYTHING_BOX = 2
    """Segment anything with box prompt."""
    SEGMENTANYTHING_VISUAL = 3
    """Segment anything with visual prompt."""
    OBB = 4
    """Draw oriented bounding box (3-click, auto-completed rectangle)."""
    BBOX = 5
    """Draw an axis-aligned bounding box (two clicks or one drag)."""


class MAPMode(Enum):
    """Canvas show map mode"""

    LABEL = 0
    SEMANTIC = 1
    INSTANCE = 2


class CONTOURMode(Enum):
    """
    Contour Mode - ways to convert masks to polygons.
    """

    SAVE_MAX_ONLY = 0
    """Only save max contour."""
    SAVE_EXTERNAL = 1
    """Only save external contour."""
    SAVE_ALL = 2
    """Only save all contour."""


class CONTOURMethod(Enum):
    """
    Contour Method - ways to convert masks to polygons.
    """
    SIMPLE = 0
    TC89_KCOS = 1
    NONE = 2
