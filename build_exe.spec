# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the portable Windows application.

The build intentionally uses one-folder mode: CUDA and Qt ship beside the
launcher, so startup does not unpack several gigabytes into a temporary
directory on every run.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "main.py").is_file():
    raise SystemExit("Run PyInstaller from the repository root.")

WINDOWS_ICON = PROJECT_ROOT / "logo" / "IVAS.ico"
VERSION_INFO = PROJECT_ROOT / "build_version_info.txt"
for required_asset in (WINDOWS_ICON, VERSION_INFO):
    if not required_asset.is_file():
        raise SystemExit("Required IVAS build asset is missing: {}".format(required_asset))

datas = collect_data_files(
    "IVAS",
    # TorchScript and Triton inspect decorated SAM functions at runtime. Keep
    # their source beside the frozen bytecode so JIT compilation can resolve it.
    include_py_files=True,
    excludes=[
        "checkpoints/**",
        "**/*.pt",
        "**/*.pth",
        "**/__pycache__/**",
        "IVAS_new_64.svg",
        "ivas_legacy_categories.yaml",
    ],
)
binaries = []
hiddenimports = []

# Bundle the reference VLC runtime as the codec fallback for media that the Qt
# backend cannot decode. Python-vlc is imported lazily after these paths are set.
vlc_root = PROJECT_ROOT / "ref" / "Surgical_Video_Phrase_Tasks_Annotation"
if (vlc_root / "libvlc.dll").is_file():
    for filename in ("libvlc.dll", "libvlccore.dll"):
        source = vlc_root / filename
        if source.is_file():
            binaries.append((str(source), "vlc"))
    plugins_root = vlc_root / "plugins"
    if plugins_root.is_dir():
        for source in plugins_root.rglob("*"):
            if source.is_file():
                destination = Path("vlc") / source.parent.relative_to(vlc_root)
                datas.append((str(source), destination.as_posix()))

logo_dir = PROJECT_ROOT / "logo"
if logo_dir.is_dir():
    datas.append((str(logo_dir), "logo"))

for documentation in ("README.md", "README-cn.md", "LICENSE"):
    source = PROJECT_ROOT / documentation
    if source.is_file():
        datas.append((str(source), "."))

smoke_image = PROJECT_ROOT / "example" / "images" / "000000000113.jpg"
if smoke_image.is_file():
    datas.append((str(smoke_image), "_build_smoke"))

checkpoint_setting = os.environ.get("IVAS_BUNDLE_CHECKPOINT", os.environ.get("IVAT_BUNDLE_CHECKPOINT"))
if checkpoint_setting:
    checkpoint_path = Path(checkpoint_setting).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit("Bundled checkpoint does not exist: {}".format(checkpoint_path))
    datas.append((str(checkpoint_path), "IVAS/checkpoints"))

# PyInstaller's official hooks discover Torch, Torchvision and Triton binaries,
# including the CUDA DLL dependency graph. Only add non-code package data here;
# collecting every Torch submodule would also pull unrelated training,
# TensorBoard, Dask, Arrow and notebook stacks into this inference application.
for package in ("timm", "hydra", "omegaconf"):
    try:
        package_data = collect_data_files(package, include_py_files=False)
    except Exception as error:
        print("Optional collection skipped for {}: {}".format(package, error))
        continue
    datas += package_data

for package in (
    "torch",
    "torchvision",
    "timm",
    "hydra-core",
    "omegaconf",
    "ivas-sam",
):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

excluded_sam3_prefixes = (
    "IVAS.segment_any.sam3.agent",
    "IVAS.segment_any.sam3.eval",
    "IVAS.segment_any.sam3.train",
)
hiddenimports += collect_submodules(
    "IVAS.segment_any.sam3",
    filter=lambda name: not name.startswith(excluded_sam3_prefixes),
)
hiddenimports += collect_submodules("IVAS.segment_any.sam2")
hiddenimports += [
    "IVAS",
    "IVAS.main",
    "IVAS.annotation",
    "IVAS.configs",
    "IVAS.theme",
    "cv2",
    "vlc",
    "PIL._tkinter_finder",
    "pycocotools._mask",
    "torchvision.io.image",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pandas",
        "dask",
        "distributed",
        "pyarrow",
        "zarr",
        "zmq",
        "matplotlib",
        "tensorboard",
        "jupyter",
        "IPython",
        "IVAS.segment_any.sam3.agent",
        "IVAS.segment_any.sam3.eval",
        "IVAS.segment_any.sam3.train",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IVAS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(WINDOWS_ICON),
    version=str(VERSION_INFO),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="IVAS",
)
