"""Three-way project merge: independent edits survive; conflicting edits fail."""

import copy


class ProjectConflictError(RuntimeError):
    pass


_MISSING = object()
_IDENTITIES = {
    "temporal_annotations": "id", "temporal_templates": "id",
    "instances": "object_id", "spatial_frames": "frame_index",
    "tracking_runs": "id", "audit": "id",
}


def merge_projects(base, local, remote, path=()):
    if not path and all(isinstance(value, dict) for value in (base, local, remote)):
        identity = base.get("project_id")
        if identity and any(value.get("project_id") != identity for value in (local, remote)):
            raise ProjectConflictError("The project was replaced by another project")
    if local == remote or remote == base:
        return copy.deepcopy(local) if local is not _MISSING else _MISSING
    if local == base:
        return copy.deepcopy(remote) if remote is not _MISSING else _MISSING
    if path and path[-1] == "updated_at" and all(isinstance(v, str) for v in (local, remote)):
        return max(local, remote)
    if path == ("revision",) and all(isinstance(v, int) for v in (local, remote)):
        return max(local, remote)
    if all(isinstance(value, dict) for value in (base, local, remote)):
        result = {}
        for key in dict.fromkeys([*remote, *local, *base]):
            value = merge_projects(base.get(key, _MISSING), local.get(key, _MISSING),
                                   remote.get(key, _MISSING), path + (key,))
            if value is not _MISSING:
                result[key] = value
        return result
    if all(isinstance(value, list) for value in (base, local, remote)):
        identity = _IDENTITIES.get(path[0]) if len(path) == 1 else None
        if identity:
            maps = [{item[identity]: item for item in value} for value in (base, local, remote)]
            result = []
            for key in dict.fromkeys([*maps[2], *maps[1], *maps[0]]):
                value = merge_projects(*(items.get(key, _MISSING) for items in maps),
                                       path=path + (str(key),))
                if value is not _MISSING:
                    result.append(value)
            return result
        if local[:len(base)] == base and remote[:len(base)] == base:
            result = copy.deepcopy(remote)
            result.extend(copy.deepcopy(item) for item in local[len(base):] if item not in result)
            return result
    raise ProjectConflictError("Project changed concurrently at {}. Reload before retrying.".format(".".join(path)))
