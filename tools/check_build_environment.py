"""Check installed build requirements without contacting a package index."""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main(requirement_files):
    try:
        from packaging.requirements import Requirement
    except ImportError:
        print("Missing packaging; build dependencies must be installed.")
        return 1

    missing = []
    for filename in requirement_files:
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = version(requirement.name)
            except PackageNotFoundError:
                missing.append(str(requirement))
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                missing.append("{} (installed: {})".format(requirement, installed))
    if missing:
        print("Missing or incompatible build requirements:\n" + "\n".join(missing))
        return 1
    print("Runtime and build dependencies are already installed; no download needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
