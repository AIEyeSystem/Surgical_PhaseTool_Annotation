# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

"""
Startup banner for IVAS.

Prints a compact colored IVAS wordmark with version and system information.
The palette follows the supplied pink IVAS mascot logo.
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

from IVAS.branding import APP_DESCRIPTION, APP_NAME, AUTHOR


# ── Color support detection ──

def _supports_color():
    """Check if terminal supports ANSI truecolor output."""
    if os.environ.get('FORCE_COLOR'):
        return True
    if os.environ.get('NO_COLOR'):
        return False
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False

    if sys.platform == 'win32':
        # Windows Terminal
        if os.environ.get('WT_SESSION'):
            return True
        # VS Code / Cursor integrated terminal
        if os.environ.get('TERM_PROGRAM', '').lower() in ('vscode', 'cursor'):
            return True
        # Any terminal that advertises truecolor support
        if os.environ.get('COLORTERM'):
            return True
        # Git Bash / MSYS2 / MinGW
        if os.environ.get('MSYSTEM'):
            return True
        return False

    # Unix: TERM=dumb means no color
    if os.environ.get('TERM', '') == 'dumb':
        return False
    return True


# ── ANSI 24-bit color codes (matching the SVG icon) ──

class _C:
    I   = '\033[38;2;244;114;159m'
    V   = '\033[38;2;229;91;139m'
    A   = '\033[38;2;194;47;99m'
    S   = '\033[38;2;126;20;63m'
    DIM = '\033[38;2;140;140;140m'   # dim secondary text
    RST = '\033[0m'


# ── ASCII art letters — 6 rows each, figlet-style block characters ──

_L_I = [           # I — narrow vertical bar, 4 chars
    '██╗ ',
    '██║ ',
    '██║ ',
    '██║ ',
    '██║ ',
    '╚═╝ ',
]

_L_V = [           # V — converging sides, 9 chars
    '██╗   ██╗',
    '██║   ██║',
    '██║   ██║',
    '╚██╗ ██╔╝',
    ' ╚████╔╝ ',
    '  ╚═══╝  ',
]

_L_A = [           # A — triangular peak, 9 chars
    ' █████╗ ',
    '██╔══██╗',
    '███████║',
    '██╔══██║',
    '██║  ██║',
    '╚═╝  ╚═╝',
]

_L_S = [           # S — 6-row block wordmark
    '███████╗',
    '██╔════╝',
    '███████╗',
    '╚════██║',
    '███████║',
    '╚══════╝',
]


# ── Public API ──

def print_banner():
    """Print the IVAS startup banner to stdout."""

    # Ensure stdout uses UTF-8 so box-drawing characters work on Windows.
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # --- gather information ---

    try:
        from IVAS import __version__
        version = __version__
    except ImportError:
        version = 'unknown'

    author = AUTHOR
    url = ''
    desc = APP_DESCRIPTION

    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

    try:
        # Reading distribution metadata avoids importing the multi-second
        # Torch/CUDA runtime merely to decorate the startup banner. Hardware
        # availability is reported by the status monitor when it is requested.
        torch_ver = package_version('torch')
        cuda_ok = 'deferred'
    except PackageNotFoundError:
        torch_ver = 'N/A'
        cuda_ok = '-'

    # --- print ---

    use_color = _supports_color()

    if use_color:
        _print_colored(version, author, url, desc, py_ver, torch_ver, cuda_ok)
    else:
        _print_plain(version, author, url, desc, py_ver, torch_ver, cuda_ok)


# ── Internal renderers ──

def _print_colored(version, author, url, desc, py_ver, torch_ver, cuda_ok):
    """Color version: each letter in its SVG-matching color."""
    C = _C
    GAP = '  '

    print()
    # ASCII logo — 6 rows
    for i in range(6):
        line = (f'{C.I}{_L_I[i]}{C.RST}{GAP}'
                f'{C.V}{_L_V[i]}{C.RST}{GAP}'
                f'{C.A}{_L_A[i]}{C.RST}{GAP}'
                f'{C.S}{_L_S[i]}{C.RST}')
        print(f'  {line}')

    print()
    # Title line: IVAS spelled with per-letter colors
    print(f'  {C.I}I{C.RST}{C.V}V{C.RST}{C.A}A{C.RST}{C.S}S{C.RST}'
          f'  —  {APP_NAME}')
    print(f'  {C.DIM}{desc}{C.RST}')
    print()
    # Meta info
    print(f'  {C.DIM}Version{C.RST} {version}'
          f'          {C.DIM}Author{C.RST} {author}')
    print(f'  {C.DIM}License{C.RST} Apache 2.0')
    print()
    # System info
    print(f'  {C.DIM}Python{C.RST}  {py_ver}'
          f'          {C.DIM}PyTorch{C.RST} {torch_ver}'
          f'          {C.DIM}CUDA{C.RST} {cuda_ok}')
    print()


def _print_plain(version, author, url, desc, py_ver, torch_ver, cuda_ok):
    """Plain-text fallback: no ANSI codes."""
    GAP = '  '

    print()
    for i in range(6):
        line = f'{_L_I[i]}{GAP}{_L_V[i]}{GAP}{_L_A[i]}{GAP}{_L_S[i]}'
        print(f'  {line}')

    print()
    print(f'  {APP_NAME}')
    print(f'  {desc}')
    print()
    print(f'  Version {version}          Author {author}')
    print('  License Apache 2.0')
    print()
    print(f'  Python  {py_ver}          PyTorch {torch_ver}          CUDA {cuda_ok}')
    print()
    print('--------------------------------------------------------------------------------')
    print()
