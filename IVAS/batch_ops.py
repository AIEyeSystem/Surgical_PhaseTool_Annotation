# -*- coding: utf-8 -*-
"""Widget-free batch maintenance of an IVAS label directory.

Every image ``<stem>.<ext>`` keeps its objects in ``<label_root>/<stem>.json``.
This module indexes those files by category and applies recorded, undoable
batch edits to them:

* a label that cannot be read or validated is reported and never rewritten;
* objects are edited as raw JSON dictionaries, so keys IVAS does not know
  about (and legacy ISAT/IVAT descriptions) survive untouched;
* before a file changes, its original bytes are copied to
  ``<label_root>/.ivas_batch/<id>/``; every completed operation is appended to
  ``<label_root>/.ivas_batch/history.jsonl`` with its parameters, affected
  files, object counts and content hashes;
* :func:`undo_last` restores the newest operation only while every affected
  file still has the content that operation wrote.
"""

import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from IVAS.annotation import Object
from IVAS.branding import SUPPORTED_ANNOTATION_FORMATS
from IVAS.utils.persistence import atomic_write_bytes, project_write_lock

__all__ = [
    "BATCH_DIR_NAME",
    "HISTORY_FILE_NAME",
    "OP_DELETE_CATEGORY",
    "OP_RENAME_CATEGORY",
    "OP_DELETE_EMPTY",
    "OP_UNDO",
    "MUTATING_OPS",
    "BatchError",
    "LabelReadError",
    "UndoRefused",
    "LabelIndex",
    "read_label",
    "scan_label_dir",
    "delete_category",
    "rename_category",
    "delete_empty_labels",
    "history_path",
    "read_history",
    "last_undoable",
    "undo_last",
]

BATCH_DIR_NAME = ".ivas_batch"
HISTORY_FILE_NAME = "history.jsonl"

OP_DELETE_CATEGORY = "delete_category"
OP_RENAME_CATEGORY = "rename_category"
OP_DELETE_EMPTY = "delete_empty_labels"
OP_UNDO = "undo"
MUTATING_OPS = (OP_DELETE_CATEGORY, OP_RENAME_CATEGORY, OP_DELETE_EMPTY)

# Keep history lines readable when a directory holds thousands of foreign JSONs.
_MAX_RECORDED_ERRORS = 100
_LOCK_TIMEOUT = 10.0


class BatchError(Exception):
    """A batch operation could not run."""


class LabelReadError(BatchError):
    """A label file is unreadable or is not an IVAS annotation."""


class UndoRefused(BatchError):
    """Undo would overwrite changes made after the batch operation."""


# --------------------------------------------------------------------- reading
def _category_of(obj) -> str:
    # Mirrors ``Object.from_dict`` so the index matches what the canvas shows.
    return str(obj.get("category", "unknown"))


def _has_note(data) -> bool:
    return bool(str(data.get("info", {}).get("note") or "").strip())


def _validate(data, strict: bool):
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise LabelReadError("missing the 'info' object")
    description = data["info"].get("description", "")
    if description not in SUPPORTED_ANNOTATION_FORMATS:
        raise LabelReadError("not an IVAS annotation (description={!r})".format(description))
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise LabelReadError("'objects' must be a list")
    for position, obj in enumerate(objects, 1):
        if not isinstance(obj, dict):
            raise LabelReadError("object #{} is not a JSON object".format(position))
        if strict:
            try:
                Object.from_dict(obj)
            except (TypeError, ValueError) as error:
                raise LabelReadError("object #{}: {}".format(position, error)) from None
    if strict:
        for name in ("width", "height", "depth"):
            value = data["info"].get(name)
            if value is None:
                continue
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0):
                raise LabelReadError("invalid image {} {!r}".format(name, value))


def read_label(path, strict: bool = False):
    """Return ``(data, raw_bytes)`` for an IVAS label or raise LabelReadError.

    ``strict`` additionally applies the canvas loader's per-object checks; batch
    edits use it so they never rewrite a file the application cannot open.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise LabelReadError("cannot read file: {}".format(error)) from None
    try:
        # Same decoding as Annotation.load_annotation (UTF-8, BOM rejected).
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise LabelReadError("invalid JSON: {}".format(error)) from None
    _validate(data, strict)
    return data, raw


def _category_counts(data) -> dict:
    counts = {}
    for obj in data.get("objects", []):
        category = _category_of(obj)
        counts[category] = counts.get(category, 0) + 1
    return counts


def _list_labels(root):
    """Map ``normcase(file name)`` to ``(name, mtime_ns, size)`` for top-level JSONs."""
    listing = {}
    with os.scandir(root) as entries:
        for entry in entries:
            name = entry.name
            if name.startswith(".") or not name.lower().endswith(".json"):
                continue
            try:
                if not entry.is_file():
                    continue
                stat = entry.stat()
            except OSError:
                continue
            listing[os.path.normcase(name)] = (name, stat.st_mtime_ns, stat.st_size)
    return listing


def _same_root(first, second) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


class LabelIndex:
    """Category counts per image stem for one label directory.

    Attributes:
        root: absolute label directory.
        stems: the image stems that were requested (or found, when unrestricted).
        counts: ``stem -> {category: object count}`` for every readable label
            (an existing label without objects maps to ``{}``).
        errors: ``stem -> message`` for labels that could not be read.
        signatures: ``stem -> (file name, mtime_ns, size)`` used to reuse
            unchanged entries on the next incremental scan.
        root_error: set when the directory itself could not be listed.
    """

    def __init__(self, root, stems=(), counts=None, errors=None, signatures=None,
                 root_error=None):
        self.root = root
        self.stems = tuple(stems)
        self.counts = counts or {}
        self.errors = errors or {}
        self.signatures = signatures or {}
        self.root_error = root_error

    def categories_for(self, stem) -> dict:
        return self.counts.get(stem, {})

    def has_label(self, stem) -> bool:
        return stem in self.counts or stem in self.errors

    def image_counts(self, stems=None) -> dict:
        """``category -> number of images`` holding at least one such object."""
        result = {}
        for stem in self.stems if stems is None else stems:
            for category in self.counts.get(stem, ()):
                result[category] = result.get(category, 0) + 1
        return result

    def object_counts(self, stems=None) -> dict:
        result = {}
        for stem in self.stems if stems is None else stems:
            for category, count in self.counts.get(stem, {}).items():
                result[category] = result.get(category, 0) + count
        return result

    def stems_with(self, category, stems=None) -> list:
        return [stem for stem in (self.stems if stems is None else stems)
                if category in self.counts.get(stem, ())]


def scan_label_dir(label_root, stems=None, previous=None, cancelled=None):
    """Index ``label_root``; never raises for unreadable files.

    Arguments:
        stems: restrict the scan to these image stems (``<stem>.json``). When
            omitted every non-hidden top-level ``*.json`` is indexed.
        previous: an earlier :class:`LabelIndex` of the same directory. Entries
            whose name, mtime and size are unchanged are reused without parsing,
            so a rescan after a save costs one directory listing.
        cancelled: optional ``threading.Event``; returns ``None`` once set.
    """
    root = os.path.abspath(os.fspath(label_root))
    try:
        listing = _list_labels(root)
    except OSError as error:
        return LabelIndex(root, stems or (), root_error=str(error))
    if stems is None:
        wanted = sorted(os.path.splitext(name)[0] for name, _, _ in listing.values())
    else:
        wanted = list(dict.fromkeys(stems))
    reuse = previous if previous is not None and _same_root(previous.root, root) else None
    counts, errors, signatures = {}, {}, {}
    for stem in wanted:
        if cancelled is not None and cancelled.is_set():
            return None
        item = listing.get(os.path.normcase(stem + ".json"))
        if item is None:
            continue
        signatures[stem] = item
        if reuse is not None and reuse.signatures.get(stem) == item:
            if stem in reuse.counts:
                counts[stem] = reuse.counts[stem]
                continue
            if stem in reuse.errors:
                errors[stem] = reuse.errors[stem]
                continue
        try:
            data, _ = read_label(os.path.join(root, item[0]))
        except LabelReadError as error:
            errors[stem] = str(error)
        else:
            counts[stem] = _category_counts(data)
    return LabelIndex(root, wanted, counts, errors, signatures)


# ------------------------------------------------------------------- mutation
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(data) -> bytes:
    # Same layout as Annotation.save_annotation.
    return json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8")


def _require_root(label_root) -> str:
    if not label_root:
        raise BatchError("No label directory is open")
    root = os.path.abspath(os.fspath(label_root))
    if not os.path.isdir(root):
        raise BatchError("Label directory does not exist: {}".format(root))
    return root


def history_path(label_root) -> str:
    return os.path.join(os.path.abspath(os.fspath(label_root)), BATCH_DIR_NAME, HISTORY_FILE_NAME)


def _batch_lock(root):
    # Serialises batch edits and undo across threads and processes.
    return project_write_lock(history_path(root), timeout=_LOCK_TIMEOUT)


def _candidates(root, stems):
    """Return ``(file name, mtime_ns)`` of the existing labels in scope."""
    listing = _list_labels(root)
    if stems is None:
        items = list(listing.values())
    else:
        items = [listing[key] for key in dict.fromkeys(
            os.path.normcase(str(stem) + ".json") for stem in stems) if key in listing]
    return sorted((name, mtime_ns) for name, mtime_ns, _ in items)


def _throttled(progress, interval=0.1):
    """Wrap ``progress(done, total)`` so callers may report every file cheaply."""
    if progress is None:
        return lambda done, total: None
    last = [0.0]

    def report(done, total):
        now = time.monotonic()
        if done >= total or now - last[0] >= interval:
            last[0] = now
            progress(done, total)
    return report


def _plan(root, stems, transform, report):
    changes, errors, scanned = [], [], 0
    candidates = _candidates(root, stems)
    # Reading counts as the first half of the work, writing as the second.
    total = 2 * len(candidates)
    for position, (name, mtime_ns) in enumerate(candidates, 1):
        report(position, total)
        path = os.path.join(root, name)
        try:
            data, raw = read_label(path, strict=True)
        except LabelReadError as error:
            errors.append({"file": name, "error": str(error)})
            continue
        scanned += 1
        change = transform(data)
        if change is not None:
            change.update(file=name, path=path, raw=raw, mtime_ns=mtime_ns)
            changes.append(change)
    return changes, errors, scanned


def _file_state(payload: bytes, path) -> dict:
    stat = os.stat(path)
    return {"sha256": _sha256(payload), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_backup(path: Path, data: bytes):
    # The backup directory is new, so no reader can see a partial file; skip
    # the temp-file rename but make the copy durable before labels change.
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _unique_dir(base: Path) -> Path:
    candidate, suffix = base, 1
    while candidate.exists():
        suffix += 1
        candidate = base.with_name("{}-{}".format(base.name, suffix))
    candidate.mkdir(parents=True)
    return candidate


def _append_history(root, record):
    path = Path(history_path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
    with open(path, "ab+") as stream:
        # Repair a line cut short by an earlier crash before appending.
        if stream.seek(0, os.SEEK_END) > 0:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                line = b"\n" + line
        stream.seek(0, os.SEEK_END)
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _summary(op, params, scope, stems, changes, errors, scanned, dry_run):
    return {
        "op": op,
        "params": params,
        "scope": scope,
        "scope_images": None if stems is None else len(set(stems)),
        "dry_run": dry_run,
        "scanned": scanned,
        "images": len(changes),
        "objects": sum(change["objects_affected"] for change in changes),
        "files": [{
            "file": change["file"],
            "action": "deleted" if change["data"] is None else "modified",
            "objects_before": change["objects_before"],
            "objects_after": change["objects_after"],
            "objects_affected": change["objects_affected"],
        } for change in changes],
        "errors": errors[:_MAX_RECORDED_ERRORS],
        "error_count": len(errors),
    }


def _run(label_root, op, params, stems, transform, dry_run, scope, progress):
    root = _require_root(label_root)
    report = _throttled(progress)
    if dry_run:
        changes, errors, scanned = _plan(root, stems, transform, report)
        report(1, 1)
        return _summary(op, params, scope, stems, changes, errors, scanned, True)

    with _batch_lock(root):
        changes, errors, scanned = _plan(root, stems, transform, report)
        result = _summary(op, params, scope, stems, changes, errors, scanned, False)
        result.update(id=None, record=None)
        if not changes:
            return result
        stamp = datetime.now()
        op_id = "{}-{}".format(stamp.strftime("%Y%m%d-%H%M%S-%f"), op)
        backup_dir = _unique_dir(Path(root) / BATCH_DIR_NAME / op_id)
        # Every backup exists before the first label changes.
        try:
            for change in changes:
                _write_backup(backup_dir / change["file"], change["raw"])
        except OSError as error:
            shutil.rmtree(backup_dir, ignore_errors=True)
            raise BatchError("Could not write the backup, nothing was changed: {}".format(
                error)) from None

        applied = []
        read_steps = result["scanned"] + len(errors)
        total = read_steps + len(changes)
        try:
            for position, change in enumerate(changes, 1):
                report(read_steps + position, total)
                path = change["path"]
                try:
                    if Path(path).read_bytes() != change["raw"]:
                        raise BatchError("changed while the operation was running")
                    if change["data"] is None:
                        os.remove(path)
                        after = None
                    else:
                        payload = _dump(change["data"])
                        atomic_write_bytes(path, payload)
                        after = _file_state(payload, path)
                except (OSError, BatchError) as error:
                    errors.append({"file": change["file"], "error": str(error)})
                    continue
                applied.append({
                    "file": change["file"],
                    "action": "deleted" if after is None else "modified",
                    "objects_before": change["objects_before"],
                    "objects_after": change["objects_after"],
                    "objects_affected": change["objects_affected"],
                    "before": {"sha256": _sha256(change["raw"]), "size": len(change["raw"]),
                               "mtime_ns": change["mtime_ns"]},
                    "after": after,
                })
        finally:
            if applied:
                record = {
                    "id": op_id,
                    "timestamp": stamp.isoformat(timespec="seconds"),
                    "op": op,
                    "params": params,
                    "scope": scope,
                    "scope_images": result["scope_images"],
                    "backup_dir": Path(os.path.relpath(backup_dir, root)).as_posix(),
                    "images": len(applied),
                    "objects": sum(item["objects_affected"] for item in applied),
                    "files": applied,
                    "errors": errors[:_MAX_RECORDED_ERRORS],
                    "error_count": len(errors),
                    "status": "partial" if len(applied) < len(changes) else "done",
                    "undone": False,
                }
                _append_history(root, record)
                result.update(id=op_id, record=record, images=record["images"],
                              objects=record["objects"], files=applied)
            else:
                shutil.rmtree(backup_dir, ignore_errors=True)
            result.update(errors=errors[:_MAX_RECORDED_ERRORS], error_count=len(errors))
    return result


def _change(data, objects, new_objects, affected, delete_file=False):
    if delete_file:
        new_data = None
    else:
        new_data = dict(data)
        new_data["objects"] = new_objects
    return {"data": new_data, "objects_before": len(objects),
            "objects_after": len(new_objects), "objects_affected": affected}


def _check_stems(stems):
    if stems is None:
        return None
    if isinstance(stems, str):
        raise TypeError("stems must be a collection of image stems, not a string")
    return list(stems)


def delete_category(label_root, category, stems=None, remove_empty_files=False,
                    dry_run=False, scope=None, progress=None):
    """Delete every object of ``category``.

    Arguments:
        stems: limit the edit to these image stems; ``None`` means every label.
        remove_empty_files: delete a label file left with no objects and no
            image note. Default ``False`` keeps the (empty) file.
        dry_run: only report what would change.
        scope: free-form scope name stored in the history record.
        progress: optional ``callable(done, total)``, called from this thread.
    Returns a summary dict (``images``/``objects`` affected, ``files``,
    ``errors``, and for real runs the history ``id``/``record``).
    """
    category = str(category)
    if not category:
        raise BatchError("Choose a category to delete")

    def transform(data):
        objects = data.get("objects", [])
        kept = [obj for obj in objects if _category_of(obj) != category]
        removed = len(objects) - len(kept)
        if not removed:
            return None
        delete_file = bool(remove_empty_files and not kept and not _has_note(data))
        return _change(data, objects, kept, removed, delete_file)

    params = {"category": category, "remove_empty_files": bool(remove_empty_files)}
    return _run(label_root, OP_DELETE_CATEGORY, params, _check_stems(stems), transform,
                dry_run, scope, progress)


def rename_category(label_root, old, new, stems=None, dry_run=False, scope=None,
                    progress=None):
    """Rename category ``old`` to ``new``; an existing ``new`` is merged into."""
    old, new = str(old), str(new).strip()
    if not old or not new:
        raise BatchError("Both the source and target category are required")
    if old == new:
        raise BatchError("The target category equals the source category")

    def transform(data):
        objects = data.get("objects", [])
        renamed = sum(1 for obj in objects if _category_of(obj) == old)
        if not renamed:
            return None
        new_objects = [dict(obj, category=new) if _category_of(obj) == old else obj
                       for obj in objects]
        return _change(data, objects, new_objects, renamed)

    return _run(label_root, OP_RENAME_CATEGORY, {"old": old, "new": new},
                _check_stems(stems), transform, dry_run, scope, progress)


def delete_empty_labels(label_root, stems=None, dry_run=False, scope=None, progress=None):
    """Delete label files that hold no objects and no image note."""

    def transform(data):
        objects = data.get("objects", [])
        if objects or _has_note(data):
            return None
        return _change(data, objects, [], 0, delete_file=True)

    return _run(label_root, OP_DELETE_EMPTY, {}, _check_stems(stems), transform,
                dry_run, scope, progress)


# -------------------------------------------------------------- history / undo
def _history_lines(root):
    """Return ``[(raw line, record or None)]``; unparsable lines are kept verbatim."""
    path = Path(history_path(root))
    if not path.is_file():
        return []
    lines = []
    for raw in path.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            record = None
        lines.append((raw, record if isinstance(record, dict) else None))
    return lines


def read_history(label_root) -> list:
    """Return the readable history records, oldest first."""
    if not label_root:
        return []
    return [record for _, record in _history_lines(os.path.abspath(os.fspath(label_root)))
            if record is not None]


def _undoable(record) -> bool:
    return (record.get("op") in MUTATING_OPS and not record.get("undone")
            and bool(record.get("files")))


def last_undoable(label_root):
    """Return the newest record :func:`undo_last` would revert, or ``None``."""
    for record in reversed(read_history(label_root)):
        if _undoable(record):
            return record
    return None


def _undo_problems(root, record, backup_dir):
    """Return ``(problems, payloads)``; payloads maps file name -> backup bytes."""
    problems, payloads = [], {}
    for item in record["files"]:
        name = item["file"]
        before = (item.get("before") or {}).get("sha256")
        try:
            payload = (backup_dir / name).read_bytes()
        except OSError:
            problems.append("{}: backup is missing".format(name))
            continue
        if _sha256(payload) != before:
            problems.append("{}: backup does not match the recorded content".format(name))
            continue
        payloads[name] = payload
        path = Path(root) / name
        try:
            current = _sha256(path.read_bytes())
        except FileNotFoundError:
            current = None
        except OSError as error:
            problems.append("{}: cannot read ({})".format(name, error))
            continue
        after = (item.get("after") or {}).get("sha256")
        if current == before:
            continue  # already back to its original content
        if current != after:
            if after is None:
                problems.append("{}: was created again after the operation".format(name))
            elif current is None:
                problems.append("{}: was deleted after the operation".format(name))
            else:
                problems.append("{}: was modified after the operation".format(name))
    return problems, payloads


def undo_last(label_root, progress=None) -> dict:
    """Restore the files changed by the newest not-yet-undone batch operation.

    Content hashes recorded in the history decide whether a file is unchanged
    since the operation (a touched-but-identical file is fine). If any file was
    edited, deleted or re-created afterwards, nothing is restored and
    :class:`UndoRefused` explains which files block the undo. On success the
    original record is marked ``undone`` and an ``undo`` record is appended.
    Returns the appended ``undo`` record.
    """
    root = _require_root(label_root)
    report = _throttled(progress)
    with _batch_lock(root):
        lines = _history_lines(root)
        target = next((record for _, record in reversed(lines)
                       if record is not None and _undoable(record)), None)
        if target is None:
            raise UndoRefused("There is no batch operation to undo")
        backup_dir = Path(root) / target.get("backup_dir", "")
        problems, payloads = _undo_problems(root, target, backup_dir)
        if problems:
            shown = problems[:10]
            if len(problems) > len(shown):
                shown.append("... {} more".format(len(problems) - len(shown)))
            raise UndoRefused(
                "Cannot undo {}: {} file(s) changed after it ran, so restoring the backup "
                "would discard those edits. Backups stay in {}.\n{}".format(
                    target["id"], len(problems), backup_dir, "\n".join(shown)))

        restored, failures = [], []
        for position, item in enumerate(target["files"], 1):
            report(position, len(target["files"]))
            try:
                atomic_write_bytes(Path(root) / item["file"], payloads[item["file"]])
            except OSError as error:
                failures.append({"file": item["file"], "error": str(error)})
            else:
                restored.append(item["file"])

        stamp = datetime.now()
        if not failures:
            target["undone"] = True
            target["undone_at"] = stamp.isoformat(timespec="seconds")
        undo_record = {
            "id": "{}-{}".format(stamp.strftime("%Y%m%d-%H%M%S-%f"), OP_UNDO),
            "timestamp": stamp.isoformat(timespec="seconds"),
            "op": OP_UNDO,
            "params": {"target": target["id"], "target_op": target["op"],
                       "target_params": target.get("params", {})},
            "scope": target.get("scope"),
            "backup_dir": target.get("backup_dir"),
            "images": len(restored),
            "objects": target.get("objects", 0),
            "files": [{"file": name, "action": "restored"} for name in restored],
            "errors": failures,
            "error_count": len(failures),
            "status": "partial" if failures else "done",
        }
        rewritten = []
        for raw, record in lines:
            if record is not None and record.get("id") == target["id"]:
                raw = json.dumps(target, ensure_ascii=False).encode("utf-8")
            rewritten.append(raw)
        rewritten.append(json.dumps(undo_record, ensure_ascii=False).encode("utf-8"))
        atomic_write_bytes(history_path(root), b"\n".join(rewritten) + b"\n")
        if failures:
            raise BatchError("Undo restored {} of {} file(s); run Undo again to retry: {}".format(
                len(restored), len(target["files"]),
                "; ".join("{}: {}".format(f["file"], f["error"]) for f in failures)))
        return undo_record
