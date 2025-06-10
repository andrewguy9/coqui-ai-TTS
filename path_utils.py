

from pathlib import Path
from typing import Generator


def walk_paths(path: Path) -> Generator[Path, None, None]:
    path = Path(path)
    for c in path.rglob('*'):
        yield c

def get_file_extension(path: Path):
    return path.suffix[1:] if path.suffix else ''

def is_audio(path_str):
    ext = get_file_extension(path_str).lower()
    return ext in ('wav', 'mp3', 'ogg')

def is_normal_file(path: Path):
    return path.is_file()

def make_extension_replacer(new_ext: str):
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    def replacer(path: Path) -> Path:
        return path.with_suffix(new_ext)
    return replacer

def get_base_name(path: Path) -> str:
    return path.stem

def get_parent_path(path: Path) -> Path:
    return path.parent

def get_relative_path(context: Path, target: Path) -> Path:
    return target.relative_to(context)
