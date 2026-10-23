"""Atomic file replacement and a reentrant, cross-process project write lock."""

import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_locks = {}
_guard = threading.Lock()
_local = threading.local()


@contextmanager
def project_write_lock(path, timeout=5.0):
    """Serialize read/merge/write as one operation, including nested save calls."""
    path = Path(path).resolve()
    key = os.path.normcase(str(path))
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    if not lock.acquire(timeout=timeout):
        raise TimeoutError("Project is busy: {}".format(path))
    held = getattr(_local, "held", None)
    if held is None:
        held = _local.held = set()
    stream = None
    acquired = False
    try:
        if key in held:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.with_name("." + path.name + ".lock").open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Project is busy: {}".format(path))
                time.sleep(0.025)
        held.add(key)
        yield
    finally:
        try:
            if acquired:
                held.discard(key)
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                lock.release()
