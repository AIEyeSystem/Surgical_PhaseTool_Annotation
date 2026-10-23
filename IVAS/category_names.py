"""Display-only category aliases; annotation names always come from ivas.yaml."""

import logging
from pathlib import Path

import yaml


def load_category_aliases(labels, config_file):
    """Use a rename file only when every entry matches the active category set.

    A partial or inconsistent file is ignored as a whole. Canonical category
    names and the fallback IDs always come from the active category config.
    """
    path = Path(config_file).with_name("ivas_rename.yaml")
    fallback = {label.get("name", "UNKNOW"): str(label.get("id", index))
                for index, label in enumerate(labels)}
    if not path.is_file():
        return fallback
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        renamed = data.get("label", data) if isinstance(data, dict) else data
        return _match_all_aliases(labels, renamed)
    except (OSError, yaml.YAMLError, UnicodeError, ValueError) as error:
        logging.getLogger(__name__).warning("Ignoring entire rename file %s: %s", path, error)
        return fallback


def _match_all_aliases(labels, renamed):
    names = [label.get("name", "UNKNOW") for label in labels]
    ids = [str(label.get("id", index)) for index, label in enumerate(labels)]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("category names and IDs must be unique")
    if not isinstance(renamed, (dict, list)) or len(renamed) != len(labels):
        raise ValueError("category counts do not match")

    if isinstance(renamed, dict):
        keyed = {str(key): value for key, value in renamed.items()}
        if len(keyed) != len(renamed):
            raise ValueError("duplicate category keys")
        if set(keyed) == set(names):
            entries = [keyed[name] for name in names]
        elif set(keyed) == set(ids):
            entries = [keyed[category_id] for category_id in ids]
        else:
            raise ValueError("category keys do not match")
    elif any(isinstance(entry, dict) and "id" in entry for entry in renamed):
        if not all(isinstance(entry, dict) and "id" in entry for entry in renamed):
            raise ValueError("every rename entry must have an ID")
        keyed = {str(entry["id"]): entry for entry in renamed}
        if len(keyed) != len(renamed) or set(keyed) != set(ids):
            raise ValueError("category IDs do not match or are duplicated")
        entries = [keyed[category_id] for category_id in ids]
    else:
        if set(ids) != {str(index) for index in range(len(labels))}:
            raise ValueError("positional aliases do not cover the category IDs")
        entries = [renamed[int(category_id)] for category_id in ids]

    aliases = {}
    for index, label in enumerate(labels):
        entry = entries[index]
        if isinstance(entry, dict):
            if "id" in entry and str(entry["id"]) != ids[index]:
                raise ValueError("category ID does not match")
            if "color" in entry and "color" in label:
                if str(entry["color"]).lower() != str(label["color"]).lower():
                    raise ValueError("category colors do not match")
            entry = entry.get("name")
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("each category must have a nonempty alias")
        aliases[names[index]] = entry.strip()
    return aliases
