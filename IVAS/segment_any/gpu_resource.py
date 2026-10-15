# -*- coding: utf-8 -*-
# @Maintainer: zack zhao
"""Low-overhead GPU memory monitoring for the status bar."""

import platform

import torch
from PyQt5.QtCore import QThread, pyqtSignal


osplatform = platform.system()


class GPUResource_Thread(QThread):
    """Report device memory once per interval without spawning subprocesses."""

    message = pyqtSignal(str)

    def __init__(self, gpu_id: int = 0, interval_ms: int = 1000, parent=None):
        super(GPUResource_Thread, self).__init__(parent)
        self.gpu_id = int(gpu_id)
        self.interval_ms = max(250, int(interval_ms))

    def run(self):
        while not self.isInterruptionRequested():
            try:
                with torch.cuda.device(self.gpu_id):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    process_bytes = torch.cuda.memory_allocated()
                    used_bytes = total_bytes - free_bytes
                self.message.emit(
                    "GPU {used:.0f}/{total:.0f} MiB   IVAS {process:.0f} MiB".format(
                        used=used_bytes / (1024**2),
                        total=total_bytes / (1024**2),
                        process=process_bytes / (1024**2),
                    )
                )
            except (RuntimeError, ValueError, AssertionError):
                self.message.emit("GPU unavailable")
            self.msleep(self.interval_ms)

    def stop(self, timeout_ms: int = 2000) -> None:
        self.requestInterruption()
        if self.isRunning():
            self.wait(timeout_ms)
