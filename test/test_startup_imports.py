import subprocess
import sys
import unittest
from pathlib import Path


class StartupImportTests(unittest.TestCase):
    def _run_probe(self, probe):
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg="stdout:\n{}\nstderr:\n{}".format(result.stdout, result.stderr),
        )

    def test_importing_entrypoint_keeps_heavy_gui_and_model_modules_lazy(self):
        probe = """
import sys

import IVAS.main

forbidden = (
    "torch",
    "PyQt5.QtCore",
    "PyQt5.QtWidgets",
    "IVAS.theme",
    "IVAS.widgets.mainwindow",
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit("eager imports: " + ", ".join(loaded))
"""
        self._run_probe(probe)

    def test_sam_smoke_route_does_not_enter_the_gui_startup_path(self):
        probe = """
import sys

import IVAS.main as entrypoint

entrypoint._run_frozen_sam31_smoke = lambda: 17
sys.argv.append("--build-sam31-smoke-test")
result = entrypoint.main()
if result != 17:
    raise SystemExit("unexpected smoke result: {}".format(result))

forbidden = ("PyQt5.QtCore", "PyQt5.QtWidgets", "IVAS.widgets.mainwindow")
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit("GUI imports on SAM smoke route: " + ", ".join(loaded))
"""
        self._run_probe(probe)

    def test_portable_build_runs_the_frozen_fused_workbench_smoke(self):
        repository = Path(__file__).resolve().parents[1]
        flag = "--build-video-workbench-smoke-test"

        build_script = (repository / "build_portable_exe.ps1").read_text(
            encoding="utf-8-sig"
        )
        launcher = (repository / "main.py").read_text(encoding="utf-8-sig")
        entrypoint = (repository / "IVAS" / "main.py").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn(flag, build_script)
        self.assertIn("VIDEO_WORKBENCH_SMOKE passed", entrypoint)
        self.assertIn(flag, entrypoint)
        self.assertIn(flag, launcher)

    def test_startup_banner_reports_metadata_without_importing_torch(self):
        probe = """
import contextlib
import io
import sys

from IVAS.banner import print_banner

with contextlib.redirect_stdout(io.StringIO()):
    print_banner()
if "torch" in sys.modules:
    raise SystemExit("startup banner imported torch")
"""
        self._run_probe(probe)

    def test_mainwindow_module_keeps_model_and_video_backends_lazy(self):
        probe = """
import sys

import IVAS.widgets.mainwindow

forbidden = (
    "torch",
    "cv2",
    "pydicom",
    "requests",
    "imgviz",
    "skimage",
    "fuzzywuzzy",
    "IVAS.segment_any.segment_any",
    "IVAS.segment_any.gpu_resource",
    "IVAS.widgets.video_annotation_workspace",
    "IVAS.widgets.converter_dialog",
    "IVAS.widgets.auto_segment_dialog",
    "IVAS.widgets.video_to_frames_dialog",
    "IVAS.widgets.annos_validator_dialog",
    "IVAS.widgets.process_exif_dialog",
    "IVAS.widgets.shortcut_dialog",
    "IVAS.widgets.about_dialog",
    "IVAS.widgets.category_setting_dialog",
    "IVAS.widgets.category_edit_dialog",
    "IVAS.widgets.model_manager_dialog",
    "IVAS.widgets.remote_sam_dialog",
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit("eager workbench backends: " + ", ".join(loaded))
"""
        self._run_probe(probe)


if __name__ == "__main__":
    unittest.main()
