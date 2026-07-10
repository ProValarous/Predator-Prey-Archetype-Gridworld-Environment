# src/baselines/registry/algorithm_registry.py

_ALGORITHMS = {}


def register(name: str, cls):
    if name in _ALGORITHMS:
        raise ValueError(f"Algorithm '{name}' already registered.")
    _ALGORITHMS[name] = cls


def get(name: str):
    if name not in _ALGORITHMS:
        raise ValueError(f"Algorithm '{name}' not registered.")
    return _ALGORITHMS[name]


def list_algorithms():
    return list(_ALGORITHMS.keys())
