import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from IVAS.widgets import model_manager_dialog as download


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body)), **headers}

    def getcode(self):
        return self.status


class DownloadIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tmp").mkdir()
        self.worker = download.DownloadThread()
        self.worker.setNameAndUrl("weights.pt", ["https://example.invalid/weights"])
        self.signals = []
        self.worker.tag.connect(lambda current, total: self.signals.append((current, total)))
        patch = mock.patch.object(download, "CHECKPOINT_PATH", str(self.root))
        patch.start(); self.addCleanup(patch.stop)

    def run_response(self, response):
        with mock.patch.object(download.request, "urlopen", return_value=response):
            self.worker.run()

    def test_truncated_body_is_never_published(self):
        self.run_response(Response(b"abc", **{"Content-Length": "10"}))
        self.assertFalse((self.root / "weights.pt").exists())
        self.assertEqual((self.root / "tmp/weights.pt").read_bytes(), b"abc")
        self.assertNotIn((-2, -2), self.signals)
        self.assertIn("incomplete", self.worker.last_error)

    def test_success_checks_hash_and_publishes_atomically(self):
        self.worker.sha256 = hashlib.sha256(b"abc").hexdigest()
        self.run_response(Response(b"abc"))
        self.assertEqual((self.root / "weights.pt").read_bytes(), b"abc")
        self.assertFalse((self.root / "tmp/weights.pt").exists())
        self.assertIn((-2, -2), self.signals)

    def test_bad_hash_preserves_previous_checkpoint(self):
        (self.root / "weights.pt").write_bytes(b"previous")
        self.worker.sha256 = "0" * 64
        self.run_response(Response(b"abc"))
        self.assertEqual((self.root / "weights.pt").read_bytes(), b"previous")
        self.assertNotIn((-2, -2), self.signals)

    def test_valid_partial_response_is_resumed(self):
        (self.root / "tmp/weights.pt").write_bytes(b"abc")
        self.run_response(Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"}))
        self.assertEqual((self.root / "weights.pt").read_bytes(), b"abcdef")

    def test_ignored_range_restarts_instead_of_appending(self):
        (self.root / "tmp/weights.pt").write_bytes(b"old")
        self.run_response(Response(b"abcdef"))
        self.assertEqual((self.root / "weights.pt").read_bytes(), b"abcdef")

    def test_inconsistent_range_is_rejected(self):
        (self.root / "tmp/weights.pt").write_bytes(b"abc")
        self.run_response(Response(b"def", 206, **{"Content-Range": "bytes 1-3/6"}))
        self.assertFalse((self.root / "weights.pt").exists())
        self.assertEqual((self.root / "tmp/weights.pt").read_bytes(), b"abc")

    def test_416_restarts_full_download(self):
        (self.root / "tmp/weights.pt").write_bytes(b"old")
        error = HTTPError("https://example.invalid/weights", 416, "range", {}, None)
        with mock.patch.object(download.request, "urlopen", side_effect=[error, Response(b"new")]):
            self.worker.run()
        self.assertEqual((self.root / "weights.pt").read_bytes(), b"new")

    def test_pause_preserves_partial_without_completion(self):
        response = Response(b"abcdef")
        read = response.read
        def pause_after_read(size):
            result = read(size)
            self.worker.pause = True
            return result
        response.read = pause_after_read
        self.run_response(response)
        self.assertFalse((self.root / "weights.pt").exists())
        self.assertNotIn((-2, -2), self.signals)


if __name__ == "__main__":
    unittest.main()
