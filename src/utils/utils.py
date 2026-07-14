from pathlib import Path

def get_root_dir() -> Path:
    """Returns the root directory of the project"""
    # .parent is 'utils', the parent of that is 'src' the parent of that is 'ROOT'
    return Path(__file__).parent.parent.parent

def from_project_path(rel_path: str) -> Path:
    """Returns the absolute path of the relative path from the project directory"""
    return get_root_dir() / rel_path