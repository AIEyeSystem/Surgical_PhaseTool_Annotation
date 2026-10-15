import os
from enum import Enum

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
SOFTWARE_CONFIG_FILE = os.path.join(IVAS_ROOT, "software.yaml")
"""Software config file - IVAS/software.yaml"""
CONFIG_FILE = os.path.join(IVAS_ROOT, CATEGORY_CONFIG_NAME)
"""Default IVAS category configuration file."""
CHECKPOINT_PATH = os.path.join(IVAS_ROOT, "checkpoints")
"""Checkpoints save root - IVAS/checkpoints"""

os.makedirs(os.path.join(CHECKPOINT_PATH, "tmp"), exist_ok=True)

if not os.path.exists(SOFTWARE_CONFIG_FILE):
    with open(SOFTWARE_CONFIG_FILE, "w") as f:
        pass

if not os.path.exists(CONFIG_FILE):
    legacy_config = next(
        (os.path.join(IVAS_ROOT, name) for name in LEGACY_CATEGORY_CONFIG_NAMES
         if os.path.isfile(os.path.join(IVAS_ROOT, name))),
        None,
    )
    with open(CONFIG_FILE, "wb") as f:
        if legacy_config is not None:
            with open(legacy_config, "rb") as source:
                f.write(source.read())


def load_config(file: str) -> dict:
    r"""
    Load config file

    Arguments:
        file (str): config file path

    Returns:
        dict: config dict
    """
    with open(file, "rb") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg


def save_config(cfg: dict, file: str) -> None:
    """
    Save config file

    Arguments:
        cfg (dict): config dict
        file (str): config file path
    """
    s = yaml.dump(cfg, allow_unicode=True)
    with open(file, "w", encoding="utf-8") as f:
        f.write(s)


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
