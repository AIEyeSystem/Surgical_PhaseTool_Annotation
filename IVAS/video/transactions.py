"""Recoverable label changes coordinated with the atomic project manifest."""

import copy
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from IVAS.utils.persistence import atomic_write_bytes, project_write_lock


def _path(store, value):
    path = Path(value)
    return path if path.is_absolute() else store.case_directory / path


def _restore(store, entries):
    # Read every backup before modifying even the first accepted label.
    contents = [(_path(store, item["path"]),
                 _path(store, item["backup"]).read_bytes() if item["existed"] else None)
                for item in entries]
    for path, data in contents:
        if data is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, data)


def _cleanup(store, journal, entries):
    # A leftover journal after a successful commit is harmless: recovery checks
    # the manifest before deciding whether to restore preimages.
    try:
        journal.unlink(missing_ok=True)
    except OSError:
        return
    for item in entries:
        if item.get("backup"):
            try:
                _path(store, item["backup"]).unlink(missing_ok=True)
            except OSError:
                pass


def recover_label_transactions(store):
    with project_write_lock(store.path):
        for journal in sorted(store.tracking_directory.glob(".transaction-*.json")):
            data = json.loads(journal.read_text(encoding="utf-8"))
            try:
                current = type(store)._load_file(store.path).project
            except (OSError, ValueError, KeyError, TypeError):
                if not store.recovered_from_backup:
                    raise
                current = store._persisted_project
            run = next((r for r in current["tracking_runs"] if r["id"] == data["run_id"]), {})
            if run.get("status") != data["target_status"]:
                _restore(store, data["entries"])
            _cleanup(store, journal, data["entries"])


@contextmanager
def label_transaction(store, run_id, target_status, paths):
    """The caller saves its final manifest inside this context, after label IO."""
    with project_write_lock(store.path):
        snapshot = store._runtime_snapshot()
        token = uuid.uuid4().hex
        entries = []
        for index, path in enumerate(dict.fromkeys(Path(p).resolve() for p in paths)):
            backup = store.tracking_directory / (".transaction-{}-{}.bak".format(token, index))
            existed = path.is_file()
            if existed:
                atomic_write_bytes(backup, path.read_bytes())
            entries.append({"path": os.path.relpath(path, store.case_directory),
                            "existed": existed,
                            "backup": os.path.relpath(backup, store.case_directory) if existed else ""})
        journal = store.tracking_directory / (".transaction-{}.json".format(token))
        atomic_write_bytes(journal, json.dumps({"run_id": run_id, "target_status": target_status,
                                               "entries": entries}).encode("utf-8"))
        try:
            yield
        except BaseException:
            store._restore_runtime_snapshot(snapshot)
            # An exception can occur after os.replace committed the manifest
            # (for example while updating in-memory bookkeeping). The durable
            # decision wins: never undo labels belonging to a committed run.
            # If the manifest cannot be read, retain the journal for recovery.
            current = type(store)._load_file(store.path).project
            run = next((item for item in current["tracking_runs"] if item["id"] == run_id), {})
            if run.get("status") == target_status:
                store.project = current
                store._persisted_project = copy.deepcopy(current)
                store.is_dirty = False
            else:
                _restore(store, entries)
            _cleanup(store, journal, entries)
            raise
        else:
            _cleanup(store, journal, entries)
