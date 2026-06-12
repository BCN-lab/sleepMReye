from pathlib import Path


def repo_root(marker=".git"):
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"no {marker} found above {p}")
