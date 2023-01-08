from pathlib import Path


def get_repo_root(filepath):
    for path in Path(filepath).parents:
        if (path / ".git").exists():
            return path
