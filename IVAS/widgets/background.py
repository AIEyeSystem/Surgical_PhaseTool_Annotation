"""Small cancellable jobs with GUI updates delivered only through Qt signals."""

from threading import Event
from PyQt5 import QtCore


_active = set()


class TaskWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    result = QtCore.pyqtSignal(object, str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancel_event = Event()
        _active.add(self)
        self.finished.connect(self._release)

    def cancel(self):
        self.cancel_event.set()
        self.requestInterruption()

    def run(self):
        try:
            result = self.operation(self.cancel_event, self.progress.emit)
            self.result.emit(result, "")
        except Exception as error:
            self.result.emit(None, str(error))

    def _release(self):
        _active.discard(self)
        self.deleteLater()


class TaskDialogMixin:
    """Keep a tool alive until its job has observed cancellation and finished."""

    def start_task(self, operation):
        if getattr(self, "_worker", None) is not None:
            return False
        self._close_when_finished = False
        self.pushButton_start.setEnabled(False)
        self.textBrowser.clear()
        self.progressBar.reset()
        worker = self._worker = TaskWorker(operation)
        worker.progress.connect(self._task_progress)
        worker.result.connect(self._task_result)
        worker.finished.connect(self._task_finished)
        worker.start()
        return True

    def _task_progress(self, current, total, message):
        self.progressBar.setMaximum(max(1, total))
        self.progressBar.setValue(current)
        if message:
            self.textBrowser.append(message)

    def _task_result(self, result, error):
        if error:
            self.textBrowser.append("Error: " + error)
        elif self._worker is not None and self._worker.cancel_event.is_set():
            self.textBrowser.append("Cancelled")
        elif result is not None:
            self.textBrowser.append(str(result))

    def _task_finished(self):
        self._worker = None
        self.pushButton_start.setEnabled(True)
        if self._close_when_finished:
            self.close()

    def closeEvent(self, event):
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.isRunning():
            self._close_when_finished = True
            worker.cancel()
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        self.close()
