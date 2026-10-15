"""Display-only category aliases; annotation names always come from ivas.yaml."""

import logging
from pathlib import Path

import yaml


def load_category_aliases(labels, config_file):
    """Match a sibling rename file by ID/name, or by the standard label order."""
    path = Path(config_file).with_name("ivas_rename.yaml")
    renamed = {}
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
            renamed = data.get("label", data) if isinstance(data, dict) else data
        except (OSError, yaml.YAMLError, UnicodeError) as error:
            logging.getLogger(__name__).warning("Cannot read %s: %s", path, error)

    aliases = {}
    for index, label in enumerate(labels):
        name = label.get("name", "UNKNOW")
        category_id = label.get("id", index)
        entry = None
        if isinstance(renamed, dict):
            entry = renamed.get(name, renamed.get(category_id, renamed.get(str(category_id))))
        elif isinstance(renamed, list):
            if any(isinstance(item, dict) and "id" in item for item in renamed):
                entry = next((item for item in renamed if isinstance(item, dict)
                              and str(item.get("id")) == str(category_id)), None)
            else:
                # Persisted IDs keep positional aliases stable after a category
                # is removed or reordered in the settings dialog.
                position = category_id if isinstance(category_id, int) else index
                if 0 <= position < len(renamed):
                    entry = renamed[position]
        if isinstance(entry, dict):
            entry = entry.get("name")
        alias = str(entry).strip() if entry is not None else ""
        aliases[name] = alias or str(category_id)
    return aliases
