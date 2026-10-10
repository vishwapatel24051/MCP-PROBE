import os
from typing import Final

ROOT_DIR: Final[str] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def repo_path(*parts: str) -> str:
    return os.path.join(ROOT_DIR, *parts)


def detectors_path(filename: str) -> str:
    return repo_path('configs', 'detectors', filename)


def servers_path(filename: str) -> str:
    return repo_path('configs', 'servers', filename)


def models_path(filename: str) -> str:
    return repo_path('configs', 'models', filename)


def learnableshield_models_path(filename: str) -> str:
    return repo_path('learnableshield_models', filename)


def exists(path: str) -> bool:
    return os.path.exists(path)

